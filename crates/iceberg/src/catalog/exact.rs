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

use std::any::{Any, type_name};
use std::fmt::{Debug, Display, Formatter};

use uuid::Uuid;

use crate::table::Table;
use crate::{Error, ErrorKind, Result, TableIdent};

/// An exact table base loaded by a catalog.
///
/// This receipt is intentionally move-only. Its token is opaque to callers and
/// can only be interpreted by the catalog implementation that created it.
pub struct ExactTableBase {
    table: Table,
    token: Box<dyn Any + Send + Sync>,
}

impl ExactTableBase {
    /// Creates an exact base with a catalog-private token.
    ///
    /// Catalog implementations use this method when implementing
    /// [`crate::Catalog::load_table_exact`]. Callers should treat the token as
    /// opaque and obtain receipts from the catalog instead.
    #[doc(hidden)]
    pub fn new<T>(table: Table, token: T) -> Self
    where T: Any + Send + Sync {
        Self {
            table,
            token: Box::new(token),
        }
    }

    /// Returns the exact table used to construct a transaction.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Consumes the receipt and extracts its catalog-private token.
    #[doc(hidden)]
    pub fn into_parts<T>(self) -> Result<(Table, T)>
    where T: Any + Send + Sync {
        let token = self.token.downcast::<T>().map_err(|_| {
            Error::new(
                ErrorKind::PreconditionFailed,
                format!(
                    "Exact table base token does not belong to the expected catalog adapter ({})",
                    type_name::<T>()
                ),
            )
        })?;

        Ok((self.table, *token))
    }
}

impl Debug for ExactTableBase {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactTableBase")
            .field("table", &self.table)
            .field("token", &"<opaque>")
            .finish()
    }
}

/// What a [`PreparedExactCommit`] is bound to, for the caller to record before
/// it sends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExactCommitBinding {
    /// The table the commit updates.
    pub table: TableIdent,
    /// The table UUID of the base and of the staged metadata.
    pub table_uuid: Uuid,
    /// The metadata location of the exact base.
    pub base_metadata_location: String,
    /// The catalog's version of the base, when it has one (Glue: `VersionId`).
    /// The send is conditioned on it.
    pub base_version_id: Option<String>,
    /// The metadata object that staging wrote and the send makes current.
    pub staged_metadata_location: String,
    /// SHA-256 of the bytes staging wrote there. The send reads the object
    /// again and refuses before its request if the bytes differ.
    pub staged_metadata_sha256: [u8; 32],
}

/// A staged exact-base commit, ready for its single catalog request.
///
/// The catalog that staged it has written the staged metadata object and sent
/// nothing. The value is move-only: only [`crate::Catalog::send_table_exact`]
/// on the staging catalog instance accepts it, and it cannot be sent twice.
///
/// ```compile_fail,E0382
/// # async fn twice(catalog: &dyn iceberg::Catalog, prepared: iceberg::PreparedExactCommit) {
/// let _ = catalog.send_table_exact(prepared).await;
/// let _ = catalog.send_table_exact(prepared).await; // use of moved value
/// # }
/// ```
pub struct PreparedExactCommit {
    binding: ExactCommitBinding,
    token: Box<dyn Any + Send + Sync>,
}

impl PreparedExactCommit {
    /// Creates a prepared commit with a catalog-private token.
    ///
    /// Catalog implementations use this method when implementing
    /// [`crate::Catalog::stage_table_exact`]. The token must hold everything
    /// the send uses; the binding is a copy for the caller.
    #[doc(hidden)]
    pub fn new<T>(binding: ExactCommitBinding, token: T) -> Self
    where T: Any + Send + Sync {
        Self {
            binding,
            token: Box::new(token),
        }
    }

    /// Returns what this commit is bound to.
    pub fn binding(&self) -> &ExactCommitBinding {
        &self.binding
    }

    /// Consumes the prepared commit and extracts its catalog-private token.
    #[doc(hidden)]
    pub fn into_parts<T>(self) -> Result<(ExactCommitBinding, T)>
    where T: Any + Send + Sync {
        let token = self.token.downcast::<T>().map_err(|_| {
            Error::new(
                ErrorKind::PreconditionFailed,
                format!(
                    "Prepared exact commit token does not belong to the expected catalog adapter ({})",
                    type_name::<T>()
                ),
            )
        })?;

        Ok((self.binding, *token))
    }
}

impl Debug for PreparedExactCommit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedExactCommit")
            .field("binding", &self.binding)
            .field("token", &"<opaque>")
            .finish()
    }
}

/// Result of committing a transaction against an exact table base.
#[derive(Debug)]
pub enum ExactCommitOutcome {
    /// The transaction produced no updates and performed no catalog mutation.
    Noop(Table),
    /// The exact catalog mutation completed successfully.
    Committed(Table),
}

/// Failure classification for an exact-base commit.
#[derive(Debug)]
pub enum ExactCommitError {
    /// The failure happened before the catalog compare-and-swap request.
    BeforeCas {
        /// Underlying action, conversion, write, or validation failure.
        source: Box<Error>,
    },
    /// The catalog explicitly rejected the conditional mutation as contended.
    ContendedNoMutation {
        /// Modeled conditional-mutation rejection from the catalog.
        source: Box<Error>,
    },
    /// The service explicitly rejected the mutation for a non-contention reason.
    RejectedNoMutation {
        /// Modeled non-contention rejection from the catalog.
        source: Box<Error>,
    },
    /// The compare-and-swap request may have been accepted.
    Ambiguous {
        /// Metadata object staged before the ambiguous catalog request.
        staged_metadata_location: String,
        /// Underlying catalog or transport failure.
        source: Box<Error>,
    },
}

impl ExactCommitError {
    /// Creates a failure known to have happened before the CAS request.
    pub fn before_cas(source: Error) -> Self {
        Self::BeforeCas {
            source: Box::new(source),
        }
    }

    /// Creates a modeled contention rejection that proves no mutation.
    pub fn contended_no_mutation(source: Error) -> Self {
        Self::ContendedNoMutation {
            source: Box::new(source),
        }
    }

    /// Creates a modeled non-contention rejection that proves no mutation.
    pub fn rejected_no_mutation(source: Error) -> Self {
        Self::RejectedNoMutation {
            source: Box::new(source),
        }
    }

    /// Creates an ambiguous result after a metadata object was staged.
    pub fn ambiguous(staged_metadata_location: String, source: Error) -> Self {
        Self::Ambiguous {
            staged_metadata_location,
            source: Box::new(source),
        }
    }

    /// Returns the underlying error.
    pub fn source_error(&self) -> &Error {
        match self {
            Self::BeforeCas { source }
            | Self::ContendedNoMutation { source }
            | Self::RejectedNoMutation { source }
            | Self::Ambiguous { source, .. } => source.as_ref(),
        }
    }
}

impl Display for ExactCommitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BeforeCas { source } => write!(f, "exact commit failed before CAS: {source}"),
            Self::ContendedNoMutation { source } => {
                write!(f, "exact commit was rejected by contention: {source}")
            }
            Self::RejectedNoMutation { source } => {
                write!(f, "exact commit was rejected without mutation: {source}")
            }
            Self::Ambiguous {
                staged_metadata_location,
                source,
            } => write!(
                f,
                "exact commit is ambiguous after staging {staged_metadata_location}: {source}"
            ),
        }
    }
}

impl std::error::Error for ExactCommitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source_error())
    }
}

/// Result returned by exact-base catalog operations.
pub type ExactCommitResult<T> = std::result::Result<T, ExactCommitError>;
