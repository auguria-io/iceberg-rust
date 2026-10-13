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

use std::sync::Arc;

use super::MemoryCatalog;
use crate::table::Table;
use crate::{
    Error, ErrorKind, ExactCommitError, ExactCommitResult, ExactTableBase, TableCommit, TableIdent,
};

#[derive(Debug)]
struct MemoryExactToken {
    catalog_identity: Arc<()>,
    metadata_location: String,
}

impl MemoryCatalog {
    pub(super) async fn load_table_exact_impl(
        &self,
        table_ident: &TableIdent,
    ) -> ExactCommitResult<ExactTableBase> {
        let root_namespace_state = self.root_namespace_state.lock().await;
        let table = self
            .load_table_from_locked_state(table_ident, &root_namespace_state)
            .await
            .map_err(before_cas)?;
        let metadata_location = table
            .metadata_location_result()
            .map_err(before_cas)?
            .to_string();

        Ok(ExactTableBase::new(table, MemoryExactToken {
            catalog_identity: Arc::clone(&self.exact_identity),
            metadata_location,
        }))
    }

    pub(super) async fn update_table_exact_impl(
        &self,
        base: ExactTableBase,
        commit: TableCommit,
    ) -> ExactCommitResult<Table> {
        let (base_table, token) = self.consume_exact_base(base, &commit)?;
        let mut root_namespace_state = self.root_namespace_state.lock().await;
        let current_metadata_location = root_namespace_state
            .get_existing_table_location(commit.identifier())
            .map_err(ExactCommitError::rejected_no_mutation)?;
        if current_metadata_location != &token.metadata_location {
            return Err(ExactCommitError::contended_no_mutation(Error::new(
                ErrorKind::CatalogCommitConflicts,
                format!(
                    "Exact base {} is no longer current for table {}",
                    token.metadata_location,
                    commit.identifier()
                ),
            )));
        }

        let staged_table = commit.apply(base_table).map_err(before_cas)?;
        staged_table
            .metadata()
            .write_to(
                staged_table.file_io(),
                staged_table
                    .metadata_location_result()
                    .map_err(before_cas)?,
            )
            .await
            .map_err(before_cas)?;

        root_namespace_state
            .commit_table_update(staged_table)
            .map_err(ExactCommitError::rejected_no_mutation)
    }

    fn consume_exact_base(
        &self,
        base: ExactTableBase,
        commit: &TableCommit,
    ) -> ExactCommitResult<(Table, MemoryExactToken)> {
        let (base_table, token) = base.into_parts::<MemoryExactToken>().map_err(before_cas)?;
        if !Arc::ptr_eq(&self.exact_identity, &token.catalog_identity) {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Exact table base belongs to a different memory catalog instance",
            )));
        }
        if commit.identifier() != base_table.identifier() {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Exact table base and commit identifiers differ",
            )));
        }

        Ok((base_table, token))
    }
}

fn before_cas(source: Error) -> ExactCommitError {
    ExactCommitError::before_cas(source)
}
