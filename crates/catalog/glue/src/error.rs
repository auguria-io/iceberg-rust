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

use std::error::Error as StdError;
use std::fmt::Debug;

use anyhow::anyhow;
use aws_sdk_glue::error::{ProvideErrorMetadata, SdkError};
use iceberg::{Error, ErrorKind};

#[derive(Debug)]
struct NoCatalogMutation(Error);

impl std::fmt::Display for NoCatalogMutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "catalog was not mutated: {}", self.0)
    }
}

impl StdError for NoCatalogMutation {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.0)
    }
}

/// Whether an ordinary Glue create/update error proves no catalog mutation.
///
/// Inspect only the error returned directly by that operation. A nested marker
/// says nothing about the outcome of an outer operation. Preparation may have
/// written an unreferenced metadata object; this is not proof of no storage I/O.
pub fn is_known_no_catalog_mutation(error: &Error) -> bool {
    error
        .source()
        .is_some_and(|source| source.is::<NoCatalogMutation>())
}

pub(crate) fn no_catalog_mutation(error: Error) -> Error {
    Error::new(error.kind(), error.message())
        .with_retryable(error.retryable())
        .with_source(NoCatalogMutation(error))
}

pub(crate) fn is_service_rejection<T: ProvideErrorMetadata>(error: &SdkError<T>) -> bool {
    error
        .raw_response()
        .is_some_and(|response| response.status().is_client_error())
        && error.as_service_error().is_some_and(|source| {
            matches!(
                source.code(),
                Some(
                    "AlreadyExistsException"
                        | "EntityNotFoundException"
                        | "InvalidInputException"
                        | "ResourceNumberLimitExceededException"
                        | "ConcurrentModificationException"
                        | "ExpiredTokenException"
                        | "AccessDeniedException"
                        | "UnrecognizedClientException"
                        | "IncompleteSignature"
                        | "NotAuthorized"
                )
            )
        })
}

pub(crate) fn mutation_error<T>(
    error: SdkError<T>,
    single_attempt: bool,
    kind: ErrorKind,
    retryable: bool,
) -> Error
where
    T: StdError + ProvideErrorMetadata + Send + Sync + 'static,
{
    let settled = matches!(&error, SdkError::ConstructionFailure(_))
        || (single_attempt && is_service_rejection(&error));
    let error = Error::new(kind, format!("aws sdk error: {error:?}"))
        .with_retryable(retryable)
        .with_source(error);
    if settled {
        no_catalog_mutation(error)
    } else {
        error
    }
}

/// Format AWS SDK error into iceberg error
pub(crate) fn from_aws_sdk_error<T>(error: aws_sdk_glue::error::SdkError<T>) -> Error
where T: Debug {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting aws sdk error".to_string(),
    )
    .with_source(anyhow!("aws sdk error: {error:?}"))
}

/// Format AWS Build error into iceberg error
pub(crate) fn from_aws_build_error(error: aws_sdk_glue::error::BuildError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Operation failed for hitting aws build error".to_string(),
    )
    .with_source(anyhow!("aws build error: {error:?}"))
}

#[cfg(test)]
mod tests {
    use aws_sdk_glue::operation::update_table::UpdateTableError;

    use super::*;

    #[test]
    fn marker_preserves_error_details_but_does_not_certify_an_outer_operation() {
        let original = Error::new(ErrorKind::CatalogCommitConflicts, "original message")
            .with_retryable(true)
            .with_source(std::io::Error::other("original source"));
        let marked = no_catalog_mutation(original);
        assert_eq!(marked.kind(), ErrorKind::CatalogCommitConflicts);
        assert_eq!(marked.message(), "original message");
        assert!(marked.retryable());
        assert!(is_known_no_catalog_mutation(&marked));
        assert!(
            marked
                .source()
                .unwrap()
                .source()
                .unwrap()
                .source()
                .unwrap()
                .is::<std::io::Error>()
        );
        let outer = Error::new(ErrorKind::Unexpected, "outer effect unknown").with_source(marked);
        assert!(!is_known_no_catalog_mutation(&outer));
    }

    #[test]
    fn construction_failure_is_settled_but_timeout_is_not() {
        for single_attempt in [false, true] {
            let before: SdkError<UpdateTableError> =
                SdkError::construction_failure(std::io::Error::other("invalid request"));
            let error = mutation_error(before, single_attempt, ErrorKind::Unexpected, false);
            assert!(is_known_no_catalog_mutation(&error));
            let timeout: SdkError<UpdateTableError> =
                SdkError::timeout_error(std::io::Error::other("ExpiredTokenException"));
            let error = mutation_error(timeout, single_attempt, ErrorKind::Unexpected, false);
            assert!(!is_known_no_catalog_mutation(&error));
            assert!(error.source().unwrap().is::<SdkError<UpdateTableError>>());
        }
    }
}
