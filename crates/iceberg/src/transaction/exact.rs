// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use super::Transaction;
use crate::table::Table;
use crate::{
    Catalog, Error, ErrorKind, ExactCommitError, ExactCommitOutcome, ExactCommitResult,
    ExactTableBase, TableCommit,
};

impl Transaction {
    /// Commits this transaction once against a move-only exact table base.
    ///
    /// This opt-in path never refreshes, rebases, backs off, or retries. The
    /// transaction must have been constructed from [`ExactTableBase::table`].
    pub async fn commit_exact_base(
        self,
        catalog: &dyn Catalog,
        base: ExactTableBase,
    ) -> ExactCommitResult<ExactCommitOutcome> {
        if !Self::matches_exact_base(&self.table, base.table()) {
            return Err(ExactCommitError::before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Transaction table does not match the exact table base",
            )));
        }

        if self.actions.is_empty() {
            return Ok(ExactCommitOutcome::Noop(self.table));
        }

        let (updates, requirements) = self
            .apply_actions()
            .await
            .map_err(ExactCommitError::before_cas)?;

        if updates.is_empty() {
            return Ok(ExactCommitOutcome::Noop(self.table));
        }

        let commit = TableCommit::builder()
            .ident(self.table.identifier().to_owned())
            .updates(updates)
            .requirements(requirements)
            .build();

        catalog
            .update_table_exact(base, commit)
            .await
            .map(ExactCommitOutcome::Committed)
    }

    fn matches_exact_base(transaction_table: &Table, base_table: &Table) -> bool {
        transaction_table.identifier() == base_table.identifier()
            && transaction_table.metadata_location() == base_table.metadata_location()
            && transaction_table.metadata() == base_table.metadata()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::catalog::MockCatalog;
    use crate::transaction::tests::{make_v2_table, setup_test_table};
    use crate::transaction::{ActionCommit, ApplyTransactionAction, TransactionAction};
    use crate::{Result, TableUpdate};

    fn exact_base(table: &Table) -> ExactTableBase {
        ExactTableBase::new(table.clone(), ())
    }

    fn exact_mock() -> MockCatalog {
        let mut catalog = MockCatalog::new();
        catalog.expect_load_table().times(0);
        catalog.expect_update_table().times(0);
        catalog
    }

    fn transaction_with_update(table: &Table) -> Transaction {
        let tx = Transaction::new(table);
        tx.update_table_properties()
            .set("test.key".to_string(), "test.value".to_string())
            .apply(tx)
            .unwrap()
    }

    struct CountingAction {
        calls: Arc<AtomicU32>,
        has_update: bool,
    }

    #[async_trait]
    impl TransactionAction for CountingAction {
        async fn commit(self: Arc<Self>, _table: &Table) -> Result<ActionCommit> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let updates = self
                .has_update
                .then(|| TableUpdate::SetLocation {
                    location: "s3://bucket/exact".to_string(),
                })
                .into_iter()
                .collect();
            Ok(ActionCommit::new(updates, Vec::new()))
        }
    }

    #[tokio::test]
    async fn true_noop_skips_all_catalog_calls() {
        let table = setup_test_table("9");
        let tx = Transaction::new(&table);
        let mut catalog = exact_mock();
        catalog.expect_update_table_exact().times(0);

        let outcome = tx
            .commit_exact_base(&catalog, exact_base(&table))
            .await
            .unwrap();

        let ExactCommitOutcome::Noop(noop_table) = outcome else {
            panic!("empty exact transaction must be a no-op");
        };
        assert_eq!(noop_table.metadata_location(), table.metadata_location());
        assert_eq!(noop_table.metadata(), table.metadata());
    }

    #[tokio::test]
    async fn empty_action_updates_are_noop() {
        let table = setup_test_table("9");
        let calls = Arc::new(AtomicU32::new(0));
        let tx = CountingAction {
            calls: Arc::clone(&calls),
            has_update: false,
        }
        .apply(Transaction::new(&table))
        .unwrap();
        let mut catalog = exact_mock();
        catalog.expect_update_table_exact().times(0);

        let outcome = tx
            .commit_exact_base(&catalog, exact_base(&table))
            .await
            .unwrap();

        assert!(matches!(outcome, ExactCommitOutcome::Noop(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn mismatched_transaction_is_rejected_before_catalog_calls() {
        let table = setup_test_table("9");
        let different_table = table
            .clone()
            .with_metadata_location("s3://bucket/test/location/metadata/v2.json".to_string());
        let tx = transaction_with_update(&different_table);
        let mut catalog = exact_mock();
        catalog.expect_update_table_exact().times(0);

        let error = tx
            .commit_exact_base(&catalog, exact_base(&table))
            .await
            .unwrap_err();

        assert!(matches!(error, ExactCommitError::BeforeCas { .. }));
    }

    #[tokio::test]
    async fn ambiguous_error_is_not_retried_despite_retry_properties() {
        let table = setup_test_table("9");
        let tx = transaction_with_update(&table);
        let mut catalog = exact_mock();
        catalog
            .expect_update_table_exact()
            .times(1)
            .returning_st(|_, _| {
                Box::pin(async {
                    Err(ExactCommitError::ambiguous(
                        "s3://bucket/staged.metadata.json".to_string(),
                        Error::new(ErrorKind::Unexpected, "lost response").with_retryable(true),
                    ))
                })
            });

        let catalog: Arc<dyn Catalog> = Arc::new(catalog);
        let error = tx
            .commit_exact_base(catalog.as_ref(), exact_base(&table))
            .await
            .unwrap_err();

        assert!(matches!(error, ExactCommitError::Ambiguous { .. }));
    }

    #[tokio::test]
    async fn typed_contention_is_not_retried() {
        let table = setup_test_table("9");
        let tx = transaction_with_update(&table);
        let mut catalog = exact_mock();
        catalog
            .expect_update_table_exact()
            .times(1)
            .returning_st(|_, _| {
                Box::pin(async {
                    Err(ExactCommitError::contended_no_mutation(
                        Error::new(ErrorKind::CatalogCommitConflicts, "exact CAS lost")
                            .with_retryable(true),
                    ))
                })
            });

        let error = tx
            .commit_exact_base(&catalog, exact_base(&table))
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            ExactCommitError::ContendedNoMutation { .. }
        ));
    }

    #[tokio::test]
    async fn action_is_applied_once() {
        let table = setup_test_table("9");
        let calls = Arc::new(AtomicU32::new(0));
        let tx = CountingAction {
            calls: Arc::clone(&calls),
            has_update: true,
        }
        .apply(Transaction::new(&table))
        .unwrap();
        let mut catalog = exact_mock();
        catalog
            .expect_update_table_exact()
            .times(1)
            .returning_st(|_, _| Box::pin(async { Ok(make_v2_table()) }));

        let outcome = tx
            .commit_exact_base(&catalog, exact_base(&table))
            .await
            .unwrap();

        assert!(matches!(outcome, ExactCommitOutcome::Committed(_)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
