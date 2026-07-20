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

//! Iceberg Glue Catalog implementation.
//!
//! To build a glue catalog with configurations
//! # Example
//!
//! ```rust, no_run
//! use std::collections::HashMap;
//!
//! use iceberg::CatalogBuilder;
//! use iceberg_catalog_glue::{GLUE_CATALOG_PROP_WAREHOUSE, GlueCatalogBuilder};
//!
//! #[tokio::main]
//! async fn main() {
//!     let catalog = GlueCatalogBuilder::default()
//!         .load(
//!             "glue",
//!             HashMap::from([(
//!                 GLUE_CATALOG_PROP_WAREHOUSE.to_string(),
//!                 "s3://warehouse".to_string(),
//!             )]),
//!         )
//!         .await
//!         .unwrap();
//! }
//! ```

#![deny(missing_docs)]

mod catalog;
mod error;
mod schema;
mod utils;
// Re-exported so callers can name the types appearing in the public signatures
// below at a guaranteed-matching SDK version. Deliberately narrow: re-exporting
// the whole SDK crate would pull all of it into this crate's semver surface.
// Callers that go on to *call* Glue still need their own `aws-sdk-glue`
// dependency for `Client`; these are the only SDK types this crate's API names.
pub use aws_sdk_glue::types::{StorageDescriptor, TableInput};
pub use catalog::*;
pub use schema::storage_descriptor_for_table;
pub use utils::{
    AWS_ACCESS_KEY_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN,
    convert_to_glue_table,
};
