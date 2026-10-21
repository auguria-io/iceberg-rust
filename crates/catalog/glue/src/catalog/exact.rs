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

use anyhow::anyhow;
use aws_config::retry::RetryConfig;
use aws_sdk_glue::operation::update_table::UpdateTableError;
use aws_sdk_glue::types::TableInput;
use iceberg::table::Table;
use iceberg::{
    Error, ErrorKind, ExactCommitBinding, ExactCommitError, ExactCommitResult, ExactTableBase,
    PreparedExactCommit, TableCommit, TableIdent,
};
use sha2::{Digest, Sha256};

use super::GlueCatalog;
use crate::utils::{convert_to_glue_table, validate_namespace};
use crate::with_catalog_id;

#[derive(Debug)]
struct GlueExactToken {
    catalog_identity: Arc<()>,
    metadata_location: String,
    version_id: String,
}

struct PreparedExactUpdate {
    catalog_identity: Arc<()>,
    binding: ExactCommitBinding,
    table_ident: TableIdent,
    table_namespace: String,
    staged_table: Table,
    staged_metadata_location: String,
    table_input: TableInput,
    version_id: String,
}

impl GlueCatalog {
    pub(super) async fn load_table_exact_impl(
        &self,
        table_ident: &TableIdent,
    ) -> ExactCommitResult<ExactTableBase> {
        let (table, version_id) = self
            .load_table_with_version_id(table_ident)
            .await
            .map_err(before_cas)?;
        let version_id = version_id
            .filter(|version_id| !version_id.trim().is_empty())
            .ok_or_else(|| {
                before_cas(Error::new(
                    ErrorKind::PreconditionFailed,
                    format!("Glue table {table_ident} has no usable VersionId"),
                ))
            })?;
        let metadata_location = table
            .metadata_location_result()
            .map_err(before_cas)?
            .to_string();

        Ok(ExactTableBase::new(table, GlueExactToken {
            catalog_identity: Arc::clone(&self.exact_identity),
            metadata_location,
            version_id,
        }))
    }

    pub(super) async fn update_table_exact_impl(
        &self,
        base: ExactTableBase,
        commit: TableCommit,
    ) -> ExactCommitResult<Table> {
        let (base_table, token) = self.consume_exact_base(base, &commit)?;
        let prepared = self.prepare_exact_update(base_table, commit, token).await?;
        self.send_exact_update(prepared).await
    }

    pub(super) async fn stage_table_exact_impl(
        &self,
        base: ExactTableBase,
        commit: TableCommit,
    ) -> ExactCommitResult<PreparedExactCommit> {
        let (base_table, token) = self.consume_exact_base(base, &commit)?;
        let prepared = self.prepare_exact_update(base_table, commit, token).await?;
        Ok(PreparedExactCommit::new(prepared.binding.clone(), prepared))
    }

    /// Checks a staged commit's binding, then makes its one request.
    pub(super) async fn send_table_exact_impl(
        &self,
        prepared: PreparedExactCommit,
    ) -> ExactCommitResult<Table> {
        let (binding, prepared) = prepared
            .into_parts::<PreparedExactUpdate>()
            .map_err(before_cas)?;
        if !Arc::ptr_eq(&self.exact_identity, &prepared.catalog_identity) {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Prepared exact commit belongs to a different Glue catalog instance",
            )));
        }
        if binding != prepared.binding {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Prepared exact commit binding was changed after staging",
            )));
        }
        let stored = prepared
            .staged_table
            .file_io()
            .new_input(&prepared.staged_metadata_location)
            .map_err(before_cas)?
            .read()
            .await
            .map_err(before_cas)?;
        if <[u8; 32]>::from(Sha256::digest(&stored)) != prepared.binding.staged_metadata_sha256 {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                format!(
                    "Staged metadata {} changed after staging",
                    prepared.staged_metadata_location
                ),
            )));
        }

        self.send_exact_update(prepared).await
    }

    fn consume_exact_base(
        &self,
        base: ExactTableBase,
        commit: &TableCommit,
    ) -> ExactCommitResult<(Table, GlueExactToken)> {
        let (base_table, token) = base.into_parts::<GlueExactToken>().map_err(before_cas)?;
        if !Arc::ptr_eq(&self.exact_identity, &token.catalog_identity) {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Exact table base belongs to a different Glue catalog instance",
            )));
        }
        if commit.identifier() != base_table.identifier()
            || base_table.metadata_location() != Some(token.metadata_location.as_str())
        {
            return Err(before_cas(Error::new(
                ErrorKind::PreconditionFailed,
                "Glue exact table base does not match its receipt",
            )));
        }

        Ok((base_table, token))
    }

    async fn prepare_exact_update(
        &self,
        base_table: Table,
        commit: TableCommit,
        token: GlueExactToken,
    ) -> ExactCommitResult<PreparedExactUpdate> {
        let table_ident = commit.identifier().clone();
        let table_namespace = validate_namespace(table_ident.namespace()).map_err(before_cas)?;
        let staged_table = commit.apply(base_table).map_err(before_cas)?;
        let staged_metadata_location = staged_table
            .metadata_location_result()
            .map_err(before_cas)?
            .to_string();
        let table_input = convert_to_glue_table(
            table_ident.name(),
            staged_metadata_location.clone(),
            staged_table.metadata(),
            staged_table.metadata().properties(),
            Some(token.metadata_location.clone()),
        )
        .map_err(before_cas)?;

        // Serialize once, so the digest covers exactly the bytes written. This
        // writes what `TableMetadata::write_to` writes.
        let bytes =
            serde_json::to_vec(staged_table.metadata()).map_err(|e| before_cas(Error::from(e)))?;
        let binding = ExactCommitBinding {
            table: table_ident.clone(),
            table_uuid: staged_table.metadata().uuid(),
            base_metadata_location: token.metadata_location,
            base_version_id: Some(token.version_id.clone()),
            staged_metadata_location: staged_metadata_location.clone(),
            staged_metadata_sha256: Sha256::digest(&bytes).into(),
        };
        staged_table
            .file_io()
            .new_output(&staged_metadata_location)
            .map_err(before_cas)?
            .write(bytes.into())
            .await
            .map_err(before_cas)?;

        Ok(PreparedExactUpdate {
            catalog_identity: token.catalog_identity,
            binding,
            table_ident,
            table_namespace,
            staged_table,
            staged_metadata_location,
            table_input,
            version_id: token.version_id,
        })
    }

    async fn send_exact_update(&self, prepared: PreparedExactUpdate) -> ExactCommitResult<Table> {
        let builder = self
            .client
            .0
            .update_table()
            .database_name(prepared.table_namespace)
            .set_skip_archive(Some(true))
            .table_input(prepared.table_input)
            .version_id(prepared.version_id);
        let builder = with_catalog_id!(builder, self.config);

        builder
            .customize()
            .config_override(
                aws_sdk_glue::config::Builder::new().retry_config(RetryConfig::disabled()),
            )
            .send()
            .await
            .map_err(|error| {
                classify_update_error(
                    error,
                    &prepared.table_ident,
                    prepared.staged_metadata_location.clone(),
                )
            })?;

        Ok(prepared.staged_table)
    }
}

fn before_cas(source: Error) -> ExactCommitError {
    ExactCommitError::before_cas(source)
}

fn classify_update_error<R>(
    error: aws_sdk_glue::error::SdkError<UpdateTableError, R>,
    table_ident: &TableIdent,
    staged_metadata_location: String,
) -> ExactCommitError
where
    R: std::fmt::Debug,
{
    let source = Error::new(
        ErrorKind::Unexpected,
        format!("Glue exact update failed for table {table_ident}"),
    )
    .with_source(anyhow!("aws sdk error: {error:?}"));

    match error.as_service_error() {
        Some(UpdateTableError::ConcurrentModificationException(_)) => {
            ExactCommitError::contended_no_mutation(source)
        }
        Some(
            UpdateTableError::AlreadyExistsException(_)
            | UpdateTableError::EntityNotFoundException(_)
            | UpdateTableError::InvalidInputException(_)
            | UpdateTableError::ResourceNumberLimitExceededException(_),
        ) => ExactCommitError::rejected_no_mutation(source),
        _ if matches!(
            &error,
            aws_sdk_glue::error::SdkError::ConstructionFailure(_)
        ) =>
        {
            ExactCommitError::before_cas(source)
        }
        _ => ExactCommitError::ambiguous(staged_metadata_location, source),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use iceberg::spec::{NestedField, PrimitiveType, Schema, TableMetadataBuilder, Type};
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{Catalog, ExactCommitOutcome, MetadataLocation, TableCreation};
    use mockito::{Matcher, Server};
    use serde_json::json;
    use tempfile::TempDir;

    use super::*;
    use crate::catalog::GlueCatalogConfig;
    use crate::{AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY};

    async fn exact_fixture(
        server: &mut Server,
        temp_dir: &TempDir,
    ) -> (GlueCatalog, TableIdent, ExactTableBase) {
        let (catalog, table_ident, metadata_location) = test_catalog(server, temp_dir).await;
        let get_table = mock_get_table(server, &table_ident, metadata_location, Some("17")).await;
        let base = catalog.load_table_exact(&table_ident).await.unwrap();
        get_table.assert_async().await;

        (catalog, table_ident, base)
    }

    async fn test_catalog(
        server: &Server,
        temp_dir: &TempDir,
    ) -> (GlueCatalog, TableIdent, String) {
        let warehouse = temp_dir.path().to_string_lossy().into_owned();
        let props = HashMap::from([
            (AWS_ACCESS_KEY_ID.to_string(), "access-key".to_string()),
            (AWS_SECRET_ACCESS_KEY.to_string(), "secret-key".to_string()),
            (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
        ]);
        let catalog = GlueCatalog::new(GlueCatalogConfig {
            name: Some("exact-test".to_string()),
            uri: Some(server.url()),
            catalog_id: None,
            warehouse: warehouse.clone(),
            props,
        })
        .await
        .unwrap();
        let table_ident = TableIdent::from_strs(["db", "table"]).unwrap();
        let table_location = format!("{warehouse}/table");
        let schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let metadata = TableMetadataBuilder::from_table_creation(
            TableCreation::builder()
                .name(table_ident.name().to_string())
                .location(table_location.clone())
                .schema(schema)
                .build(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata;
        let metadata_location =
            MetadataLocation::new_with_table_location(table_location).to_string();
        metadata
            .write_to(&catalog.file_io(), &metadata_location)
            .await
            .unwrap();

        (catalog, table_ident, metadata_location)
    }

    async fn mock_get_table(
        server: &mut Server,
        table_ident: &TableIdent,
        metadata_location: String,
        version_id: Option<&str>,
    ) -> mockito::Mock {
        let mut table = json!({
            "Name": table_ident.name(),
            "DatabaseName": "db",
            "Parameters": { "metadata_location": metadata_location }
        });
        if let Some(version_id) = version_id {
            table["VersionId"] = json!(version_id);
        }

        server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.GetTable")
            .with_status(200)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(json!({ "Table": table }).to_string())
            .expect(1)
            .create_async()
            .await
    }

    fn exact_transaction(base: &ExactTableBase) -> Transaction {
        exact_transaction_on(base.table())
    }

    fn exact_transaction_on(table: &Table) -> Transaction {
        let tx = Transaction::new(table);
        tx.update_table_properties()
            .set("exact-key".to_string(), "value".to_string())
            .apply(tx)
            .unwrap()
    }

    #[tokio::test]
    async fn update_uses_receipt_without_second_get() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, _, base) = exact_fixture(&mut server, &temp_dir).await;
        let base_location = base.table().metadata_location().unwrap().to_string();
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .match_body(Matcher::Regex(r#"\"VersionId\":\"17\""#.to_string()))
            .match_body(Matcher::Regex(format!(
                r#"\"previous_metadata_location\":\"{base_location}\""#
            )))
            .with_status(200)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body("{}")
            .expect(1)
            .create_async()
            .await;

        let outcome = exact_transaction(&base)
            .commit_exact_base(&catalog, base)
            .await
            .unwrap();

        assert!(matches!(outcome, ExactCommitOutcome::Committed(_)));
        update.assert_async().await;
    }

    #[tokio::test]
    async fn load_rejects_missing_or_empty_version_id() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, table_ident, metadata_location) = test_catalog(&server, &temp_dir).await;

        for version_id in [None, Some("  ")] {
            let get_table = mock_get_table(
                &mut server,
                &table_ident,
                metadata_location.clone(),
                version_id,
            )
            .await;
            let error = catalog.load_table_exact(&table_ident).await.unwrap_err();

            assert!(matches!(error, ExactCommitError::BeforeCas { .. }));
            get_table.assert_async().await;
        }
    }

    #[tokio::test]
    async fn update_disables_sdk_retry_for_transient_ambiguity() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, _, base) = exact_fixture(&mut server, &temp_dir).await;
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .with_status(500)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(
                json!({
                    "__type": "InternalServiceException",
                    "Message": "transient failure"
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;

        let error = exact_transaction(&base)
            .commit_exact_base(&catalog, base)
            .await
            .unwrap_err();

        let ExactCommitError::Ambiguous {
            staged_metadata_location,
            ..
        } = error
        else {
            panic!("transient response must be ambiguous");
        };
        assert!(
            catalog
                .file_io()
                .new_input(staged_metadata_location)
                .unwrap()
                .exists()
                .await
                .unwrap(),
            "ambiguous staged metadata must be preserved"
        );
        update.assert_async().await;
    }

    #[tokio::test]
    async fn explicit_rejection_is_known_no_mutation() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, _, base) = exact_fixture(&mut server, &temp_dir).await;
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .with_status(400)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(
                json!({
                    "__type": "InvalidInputException",
                    "Message": "request rejected"
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;

        let error = exact_transaction(&base)
            .commit_exact_base(&catalog, base)
            .await
            .unwrap_err();

        assert!(matches!(error, ExactCommitError::RejectedNoMutation { .. }));
        update.assert_async().await;
    }

    #[tokio::test]
    async fn receipt_is_bound_to_one_catalog_instance() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, _, base) = exact_fixture(&mut server, &temp_dir).await;
        let (other_catalog, _, _) = test_catalog(&server, &temp_dir).await;

        let error = exact_transaction(&base)
            .commit_exact_base(&other_catalog, base)
            .await
            .unwrap_err();

        assert!(matches!(error, ExactCommitError::BeforeCas { .. }));
        drop(catalog);
    }

    async fn load_base(
        server: &mut Server,
        catalog: &GlueCatalog,
        table_ident: &TableIdent,
        metadata_location: &str,
    ) -> ExactTableBase {
        let get_table = mock_get_table(
            server,
            table_ident,
            metadata_location.to_string(),
            Some("17"),
        )
        .await;
        let base = catalog.load_table_exact(table_ident).await.unwrap();
        get_table.assert_async().await;
        base
    }

    /// An `UpdateTable` mock that answers success `hits` times.
    async fn mock_update(server: &mut Server, hits: usize) -> mockito::Mock {
        server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .with_status(200)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body("{}")
            .expect(hits)
            .create_async()
            .await
    }

    fn assert_before_cas(error: ExactCommitError) {
        assert!(
            matches!(error, ExactCommitError::BeforeCas { .. }),
            "{error}"
        );
    }

    #[tokio::test]
    async fn stage_then_send_split() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, table_ident, base) = exact_fixture(&mut server, &temp_dir).await;
        let base_location = base.table().metadata_location().unwrap().to_string();
        let base_uuid = base.table().metadata().uuid();

        let no_send = mock_update(&mut server, 0).await;
        let prepared = exact_transaction(&base)
            .stage_exact_base(&catalog, base)
            .await
            .unwrap();
        no_send.assert_async().await;
        no_send.remove_async().await;

        let binding = prepared.binding().clone();
        let staged = catalog
            .file_io()
            .new_input(&binding.staged_metadata_location)
            .unwrap()
            .read()
            .await
            .expect("stage writes the staged metadata before any request");
        assert_eq!(binding, ExactCommitBinding {
            table: table_ident,
            table_uuid: base_uuid,
            base_metadata_location: base_location.clone(),
            base_version_id: Some("17".to_string()),
            staged_metadata_location: binding.staged_metadata_location.clone(),
            staged_metadata_sha256: Sha256::digest(&staged).into(),
        });
        assert_ne!(binding.staged_metadata_location, base_location);
        let staged_metadata: iceberg::spec::TableMetadata =
            serde_json::from_slice(&staged).unwrap();
        assert_eq!(staged_metadata.uuid(), base_uuid);
        assert_eq!(staged_metadata.properties()["exact-key"], "value");

        let send = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .match_body(Matcher::Regex(r#"\"VersionId\":\"17\""#.to_string()))
            .match_body(Matcher::Regex(format!(
                r#"\"metadata_location\":\"{}\""#,
                binding.staged_metadata_location
            )))
            .match_body(Matcher::Regex(format!(
                r#"\"previous_metadata_location\":\"{base_location}\""#
            )))
            .with_status(200)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body("{}")
            .expect(1)
            .create_async()
            .await;
        let committed = catalog.send_table_exact(prepared).await.unwrap();

        send.assert_async().await;
        assert_eq!(
            committed.metadata_location(),
            Some(binding.staged_metadata_location.as_str())
        );
        assert_eq!(committed.metadata().properties()["exact-key"], "value");
    }

    #[tokio::test]
    async fn prepared_commit_binding() {
        // Fails to compile if PreparedExactCommit ever becomes Clone, which
        // would allow a second send of the same staged commit.
        trait AmbiguousIfClone<A> {
            fn probe() {}
        }
        impl<T> AmbiguousIfClone<()> for T {}
        impl<T: Clone> AmbiguousIfClone<u8> for T {}
        <PreparedExactCommit as AmbiguousIfClone<_>>::probe();

        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let no_send = mock_update(&mut server, 0).await;
        let (catalog, table_ident, metadata_location) = test_catalog(&server, &temp_dir).await;
        let (other_catalog, _, _) = test_catalog(&server, &temp_dir).await;
        let stage =
            |base: ExactTableBase| exact_transaction(&base).stage_exact_base(&catalog, base);

        // Catalog instance: staged by one instance, sent through another.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let prepared = stage(base).await.unwrap();
        assert_before_cas(other_catalog.send_table_exact(prepared).await.unwrap_err());

        // Catalog instance: a base from one instance staged through another.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let error = exact_transaction(&base)
            .stage_exact_base(&other_catalog, base)
            .await
            .unwrap_err();
        assert_before_cas(error);

        // Table: a transaction for another table.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let other_table = Table::builder()
            .file_io(base.table().file_io().clone())
            .metadata(base.table().metadata_ref())
            .metadata_location(metadata_location.clone())
            .identifier(TableIdent::from_strs(["db", "other"]).unwrap())
            .build()
            .unwrap();
        let error = exact_transaction_on(&other_table)
            .stage_exact_base(&catalog, base)
            .await
            .unwrap_err();
        assert_before_cas(error);

        // Base version: the receipt's table moved off the location it was
        // loaded at.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let (table, token) = base.into_parts::<GlueExactToken>().unwrap();
        let newer = metadata_location
            .parse::<MetadataLocation>()
            .unwrap()
            .with_next_version()
            .to_string();
        let moved = Table::builder()
            .file_io(table.file_io().clone())
            .metadata(table.metadata_ref())
            .metadata_location(newer)
            .identifier(table.identifier().clone())
            .build()
            .unwrap();
        let base = ExactTableBase::new(moved.clone(), token);
        let error = exact_transaction_on(&moved)
            .stage_exact_base(&catalog, base)
            .await
            .unwrap_err();
        assert_before_cas(error);

        // Staged metadata digest: the object changed between stage and send.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let prepared = stage(base).await.unwrap();
        let staged = prepared.binding().staged_metadata_location.clone();
        let output = catalog.file_io().new_output(&staged).unwrap();
        let mut bytes = catalog
            .file_io()
            .new_input(&staged)
            .unwrap()
            .read()
            .await
            .unwrap()
            .to_vec();
        bytes.push(b' ');
        output.write(bytes.into()).await.unwrap();
        assert_before_cas(catalog.send_table_exact(prepared).await.unwrap_err());

        // Staged metadata digest: the object is gone.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let prepared = stage(base).await.unwrap();
        let staged = prepared.binding().staged_metadata_location.clone();
        catalog.file_io().delete(&staged).await.unwrap();
        assert_before_cas(catalog.send_table_exact(prepared).await.unwrap_err());

        // The public binding cannot be changed to report something else.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let prepared = stage(base).await.unwrap();
        let (mut binding, update) = prepared.into_parts::<PreparedExactUpdate>().unwrap();
        binding.staged_metadata_location = format!("{metadata_location}.other");
        let relabelled = PreparedExactCommit::new(binding, update);
        assert_before_cas(catalog.send_table_exact(relabelled).await.unwrap_err());

        // A prepared value from another adapter.
        let base = load_base(&mut server, &catalog, &table_ident, &metadata_location).await;
        let binding = stage(base).await.unwrap().binding().clone();
        let foreign = PreparedExactCommit::new(binding, ());
        assert_before_cas(catalog.send_table_exact(foreign).await.unwrap_err());

        no_send.assert_async().await;
    }

    #[tokio::test]
    async fn update_types_only_modeled_concurrency_as_contention() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, _, base) = exact_fixture(&mut server, &temp_dir).await;
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .with_status(400)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(
                json!({
                    "__type": "ConcurrentModificationException",
                    "Message": "conditional mutation lost"
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;

        let error = exact_transaction(&base)
            .commit_exact_base(&catalog, base)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            ExactCommitError::ContendedNoMutation { .. }
        ));
        update.assert_async().await;
    }
}
