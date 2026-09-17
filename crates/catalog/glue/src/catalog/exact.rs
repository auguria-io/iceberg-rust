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
    Error, ErrorKind, ExactCommitError, ExactCommitResult, ExactTableBase, TableCommit, TableIdent,
};

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
            Some(token.metadata_location),
        )
        .map_err(before_cas)?;

        staged_table
            .metadata()
            .write_to(staged_table.file_io(), &staged_metadata_location)
            .await
            .map_err(before_cas)?;

        Ok(PreparedExactUpdate {
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

fn classify_update_error(
    error: aws_sdk_glue::error::SdkError<UpdateTableError>,
    table_ident: &TableIdent,
    staged_metadata_location: String,
) -> ExactCommitError {
    let source = Error::new(
        ErrorKind::Unexpected,
        format!("Glue exact update failed for table {table_ident}"),
    )
    .with_source(anyhow!("aws sdk error: {error:?}"));

    match error.as_service_error() {
        Some(UpdateTableError::ConcurrentModificationException(_))
            if crate::error::is_service_rejection(&error) =>
        {
            ExactCommitError::contended_no_mutation(source)
        }
        _ if crate::error::is_service_rejection(&error) => {
            ExactCommitError::rejected_no_mutation(source)
        }
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
    use iceberg::{Catalog, ExactCommitOutcome, MetadataLocation, NamespaceIdent, TableCreation};
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
        test_catalog_with_request_policy(server, temp_dir, None).await
    }

    async fn test_catalog_with_request_policy(
        server: &Server,
        temp_dir: &TempDir,
        single_attempt: Option<&str>,
    ) -> (GlueCatalog, TableIdent, String) {
        let warehouse = temp_dir.path().to_string_lossy().into_owned();
        let mut props = HashMap::from([
            (AWS_ACCESS_KEY_ID.to_string(), "access-key".to_string()),
            (AWS_SECRET_ACCESS_KEY.to_string(), "secret-key".to_string()),
            (AWS_REGION_NAME.to_string(), "us-east-1".to_string()),
        ]);
        if let Some(value) = single_attempt {
            props.insert("single-attempt-requests".to_string(), value.to_string());
        }
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
                .properties([("commit.retry.num-retries".to_string(), "0".to_string())])
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

    #[tokio::test]
    async fn single_attempt_catalog_does_not_retry_delete_after_ambiguous_response() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, ident, _) =
            test_catalog_with_request_policy(&server, &temp_dir, Some("true")).await;
        let request = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.DeleteTable")
            .with_status(500)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(r#"{"__type":"InternalServiceException","Message":"unknown outcome"}"#)
            .expect(1)
            .create_async()
            .await;

        assert!(catalog.drop_table(&ident).await.is_err());
        request.assert_async().await;
    }

    #[tokio::test]
    async fn ordinary_create_marks_only_known_single_attempt_rejections() {
        for (status, code, policy, expected) in [
            (400, "ExpiredTokenException", "true", true),
            (403, "AccessDeniedException", "true", true),
            (500, "ExpiredTokenException", "true", false),
            (400, "UnknownFailure", "true", false),
            (400, "ExpiredTokenException", "false", false),
        ] {
            let mut server = Server::new_async().await;
            let dir = TempDir::new().unwrap();
            let (catalog, ident, metadata) =
                test_catalog_with_request_policy(&server, &dir, Some(policy)).await;
            let get = mock_get_table(&mut server, &ident, metadata, Some("1")).await;
            let table = catalog.load_table(&ident).await.unwrap();
            get.assert_async().await;
            let create = server
                .mock("POST", "/")
                .match_header("x-amz-target", "AWSGlue.CreateTable")
                .with_status(status)
                .with_header("content-type", "application/x-amz-json-1.1")
                .with_body(json!({"__type":code,"Message":"ExpiredTokenException"}).to_string())
                .expect(1)
                .create_async()
                .await;
            let error = catalog
                .create_table(
                    ident.namespace(),
                    TableCreation::builder()
                        .name("new_table".into())
                        .location(format!("{}/new_table", dir.path().display()))
                        .schema(table.metadata().current_schema().as_ref().clone())
                        .build(),
                )
                .await
                .unwrap_err();
            assert_eq!(
                crate::is_known_no_catalog_mutation(&error),
                expected,
                "{status} {code} {policy}: {error}"
            );
            create.assert_async().await;
        }
    }

    fn property_transaction(table: &Table) -> Transaction {
        let tx = Transaction::new(table);
        tx.update_table_properties()
            .set("test-key".into(), "value".into())
            .apply(tx)
            .unwrap()
    }

    #[tokio::test]
    async fn ordinary_update_settlement_uses_response_code_status_and_actual_retry_policy() {
        for (status, code, policy, settled, kind) in [
            (
                403,
                "ExpiredTokenException",
                "true",
                true,
                ErrorKind::Unexpected,
            ),
            (
                403,
                "UnrecognizedClientException",
                "true",
                true,
                ErrorKind::Unexpected,
            ),
            (
                400,
                "ConcurrentModificationException",
                "true",
                true,
                ErrorKind::CatalogCommitConflicts,
            ),
            (
                500,
                "ConcurrentModificationException",
                "true",
                false,
                ErrorKind::CatalogCommitConflicts,
            ),
            (
                500,
                "ExpiredTokenException",
                "true",
                false,
                ErrorKind::Unexpected,
            ),
            (400, "UnknownFailure", "true", false, ErrorKind::Unexpected),
            (
                403,
                "ExpiredTokenException",
                "false",
                false,
                ErrorKind::Unexpected,
            ),
        ] {
            let mut server = Server::new_async().await;
            let temp = TempDir::new().unwrap();
            let (catalog, ident, metadata) =
                test_catalog_with_request_policy(&server, &temp, Some(policy)).await;
            let get = server
                .mock("POST", "/")
                .match_header("x-amz-target", "AWSGlue.GetTable")
                .with_status(200)
                .with_header("content-type", "application/x-amz-json-1.1")
                .with_body(
                    json!({"Table":{"Name":"table","DatabaseName":"db","VersionId":"1",
                    "Parameters":{"metadata_location":metadata}}})
                    .to_string(),
                )
                .expect(3)
                .create_async()
                .await;
            let table = catalog.load_table(&ident).await.unwrap();
            let update = server
                .mock("POST", "/")
                .match_header("x-amz-target", "AWSGlue.UpdateTable")
                .with_status(status)
                .with_header("content-type", "application/x-amz-json-1.1")
                .with_body(json!({"__type":code,"Message":"request failure"}).to_string())
                .expect(1)
                .create_async()
                .await;
            let error = property_transaction(&table)
                .commit(&catalog)
                .await
                .unwrap_err();
            assert_eq!(
                crate::is_known_no_catalog_mutation(&error),
                settled,
                "{status} {code} {policy}: {error}"
            );
            assert_eq!(error.kind(), kind);
            assert_eq!(error.retryable(), kind == ErrorKind::CatalogCommitConflicts);
            get.assert_async().await;
            update.assert_async().await;
        }
    }

    #[tokio::test]
    async fn preparation_failures_are_settled_and_send_no_catalog_mutation() {
        let mut server = Server::new_async().await;
        let temp = TempDir::new().unwrap();
        let (catalog, ident, metadata) = test_catalog(&server, &temp).await;
        let get = mock_get_table(&mut server, &ident, metadata, Some("1")).await;
        let table = catalog.load_table(&ident).await.unwrap();
        get.assert_async().await;
        get.remove_async().await;
        let create = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.CreateTable")
            .expect(0)
            .create_async()
            .await;
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .expect(0)
            .create_async()
            .await;
        let namespace = NamespaceIdent::from_strs(["nested", "namespace"]).unwrap();
        let error = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("table".into())
                    .schema(table.metadata().current_schema().as_ref().clone())
                    .build(),
            )
            .await
            .unwrap_err();
        assert!(crate::is_known_no_catalog_mutation(&error));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = calls.clone();
        let metadata = table.metadata_location().unwrap().to_owned();
        let get = server.mock("POST", "/").match_header("x-amz-target", "AWSGlue.GetTable")
            .with_status(200).with_header("content-type", "application/x-amz-json-1.1")
            .with_body_from_request(move |_| {
                let location = if count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                    metadata.clone()
                } else { "file:///nonexistent/preparation/metadata.json".into() };
                json!({"Table":{"Name":"table","DatabaseName":"db","VersionId":"1", "Parameters":{"metadata_location":location}}}).to_string().into_bytes()
            }).expect(2).create_async().await;
        let error = property_transaction(&table)
            .commit(&catalog)
            .await
            .unwrap_err();
        assert!(crate::is_known_no_catalog_mutation(&error));
        get.assert_async().await;
        create.assert_async().await;
        update.assert_async().await;
    }

    #[tokio::test]
    async fn exact_auth_rejections_require_a_known_code_and_client_status() {
        for (status, code, settled) in [
            (403, "ExpiredTokenException", true),
            (500, "ExpiredTokenException", false),
            (400, "UnknownFailure", false),
            (500, "ConcurrentModificationException", false),
        ] {
            let mut server = Server::new_async().await;
            let temp = TempDir::new().unwrap();
            let (catalog, _ident, base) = exact_fixture(&mut server, &temp).await;
            let update = server
                .mock("POST", "/")
                .match_header("x-amz-target", "AWSGlue.UpdateTable")
                .with_status(status)
                .with_header("content-type", "application/x-amz-json-1.1")
                .with_body(json!({"__type":code}).to_string())
                .expect(1)
                .create_async()
                .await;
            let error = exact_transaction(&base)
                .commit_exact_base(&catalog, base)
                .await
                .unwrap_err();
            assert_eq!(
                matches!(error, ExactCommitError::RejectedNoMutation { .. }),
                settled,
                "{status} {code}: {error}"
            );
            assert_eq!(
                matches!(error, ExactCommitError::Ambiguous { .. }),
                !settled
            );
            update.assert_async().await;
        }
    }

    #[tokio::test]
    async fn single_attempt_catalog_covers_create_register_and_ordinary_update() {
        let mut server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (catalog, ident, metadata_location) =
            test_catalog_with_request_policy(&server, &temp_dir, Some("true")).await;
        let get = mock_get_table(&mut server, &ident, metadata_location.clone(), Some("1")).await;
        let table = catalog.load_table(&ident).await.unwrap();
        get.assert_async().await;
        get.remove_async().await;

        let create = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.CreateTable")
            .with_status(500)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(r#"{"__type":"InternalServiceException","Message":"unknown outcome"}"#)
            .expect(2)
            .create_async()
            .await;
        assert!(
            catalog
                .create_table(
                    ident.namespace(),
                    TableCreation::builder()
                        .name("new_table".into())
                        .location(format!("{}/new_table", temp_dir.path().display()))
                        .schema(table.metadata().current_schema().as_ref().clone())
                        .build()
                )
                .await
                .is_err()
        );
        assert!(
            catalog
                .register_table(&ident, metadata_location.clone())
                .await
                .is_err()
        );
        create.assert_async().await;

        let get = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.GetTable")
            .with_status(200)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(
                json!({"Table":{"Name":"table","DatabaseName":"db","VersionId":"1",
                "Parameters":{"metadata_location":metadata_location}}})
                .to_string(),
            )
            .expect(2)
            .create_async()
            .await;
        let update = server
            .mock("POST", "/")
            .match_header("x-amz-target", "AWSGlue.UpdateTable")
            .with_status(500)
            .with_header("content-type", "application/x-amz-json-1.1")
            .with_body(r#"{"__type":"InternalServiceException","Message":"unknown outcome"}"#)
            .expect(1)
            .create_async()
            .await;
        let tx = Transaction::new(&table);
        assert!(
            tx.update_table_properties()
                .set("key".into(), "value".into())
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
                .is_err()
        );
        get.assert_async().await;
        update.assert_async().await;
    }

    #[tokio::test]
    async fn single_attempt_policy_is_opt_in_and_rejects_invalid_values() {
        let server = Server::new_async().await;
        let temp_dir = TempDir::new().unwrap();
        let (default, _, _) = test_catalog(&server, &temp_dir).await;
        let (disabled, _, _) =
            test_catalog_with_request_policy(&server, &temp_dir, Some("false")).await;
        let (enabled, _, _) =
            test_catalog_with_request_policy(&server, &temp_dir, Some("true")).await;
        assert_eq!(
            default
                .client
                .0
                .config()
                .retry_config()
                .unwrap()
                .max_attempts(),
            disabled
                .client
                .0
                .config()
                .retry_config()
                .unwrap()
                .max_attempts()
        );
        assert_eq!(
            enabled
                .client
                .0
                .config()
                .retry_config()
                .unwrap()
                .max_attempts(),
            1
        );
        let invalid = GlueCatalog::new(GlueCatalogConfig {
            name: Some("invalid".into()),
            uri: None,
            catalog_id: None,
            warehouse: temp_dir.path().display().to_string(),
            props: HashMap::from([("single-attempt-requests".into(), "TRUE".into())]),
        })
        .await;
        assert_eq!(invalid.unwrap_err().kind(), iceberg::ErrorKind::DataInvalid);
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
        let tx = Transaction::new(base.table());
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
