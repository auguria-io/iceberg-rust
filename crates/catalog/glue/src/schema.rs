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

/// Property `iceberg.field.id` for `Column`
pub(crate) const ICEBERG_FIELD_ID: &str = "iceberg.field.id";
/// Property `iceberg.field.optional` for `Column`
pub(crate) const ICEBERG_FIELD_OPTIONAL: &str = "iceberg.field.optional";
/// Property `iceberg.field.current` for `Column`
pub(crate) const ICEBERG_FIELD_CURRENT: &str = "iceberg.field.current";

use std::collections::{HashMap, HashSet};

use aws_sdk_glue::types::{Column, StorageDescriptor};
use iceberg::Result;
use iceberg::spec::{PrimitiveType, Schema, SchemaVisitor, TableMetadata, visit_schema};

use crate::error::from_aws_build_error;

type GlueSchema = Vec<Column>;

/// Builds the Glue `StorageDescriptor` for an Iceberg table.
///
/// This is the same function `GlueCatalog` uses on its own write path, so a
/// descriptor built here is byte-for-byte what the catalog would write for the
/// same metadata — including on the next commit, which rebuilds the descriptor
/// from scratch rather than patching it.
///
/// Only `columns` and `location` are set, because those are the only fields
/// that survive a catalog commit; anything else would be silently dropped the
/// first time the table is updated. `location` is the table root
/// (`metadata.location()`), **not** the metadata file path — conflating the two
/// yields a wrong envelope.
///
/// Prefer [`crate::convert_to_glue_table`] when creating a Glue table: it
/// derives the descriptor and the metadata-pointer parameters from the same
/// arguments, so the two cannot disagree. Reach for this lower-level function
/// only when you need the descriptor alone.
pub fn storage_descriptor_for_table(metadata: &TableMetadata) -> Result<StorageDescriptor> {
    Ok(StorageDescriptor::builder()
        .set_columns(Some(GlueSchemaBuilder::from_iceberg(metadata)?.build()))
        .location(metadata.location().to_string())
        .build())
}

#[derive(Debug, Default)]
pub(crate) struct GlueSchemaBuilder {
    schema: GlueSchema,
    is_current: bool,
    depth: usize,
    seen_names: HashSet<String>,
}

impl GlueSchemaBuilder {
    fn from_schema(schema: &Schema) -> Result<Self> {
        let mut builder = Self {
            is_current: true,
            ..Default::default()
        };

        visit_schema(schema, &mut builder)?;

        Ok(builder)
    }

    /// Creates a new `GlueSchemaBuilder` from Iceberg table metadata.
    pub fn from_iceberg(metadata: &TableMetadata) -> Result<GlueSchemaBuilder> {
        let current_schema = metadata.current_schema();
        let mut builder = Self::from_schema(current_schema)?;

        builder.is_current = false;

        for schema in metadata.schemas_iter() {
            if schema.schema_id() == current_schema.schema_id() {
                continue;
            }

            visit_schema(schema, &mut builder)?;
        }

        Ok(builder)
    }

    /// Returns the newly converted `GlueSchema`
    pub fn build(self) -> GlueSchema {
        self.schema
    }

    /// Check if is in `StructType` while traversing schema
    fn is_inside_struct(&self) -> bool {
        self.depth > 0
    }
}

impl SchemaVisitor for GlueSchemaBuilder {
    type T = String;

    fn schema(
        &mut self,
        _schema: &iceberg::spec::Schema,
        value: Self::T,
    ) -> iceberg::Result<String> {
        Ok(value)
    }

    fn before_struct_field(&mut self, _field: &iceberg::spec::NestedFieldRef) -> Result<()> {
        self.depth += 1;
        Ok(())
    }

    fn r#struct(
        &mut self,
        r#_struct: &iceberg::spec::StructType,
        results: Vec<String>,
    ) -> iceberg::Result<String> {
        Ok(format!("struct<{}>", results.join(", ")))
    }

    fn after_struct_field(&mut self, _field: &iceberg::spec::NestedFieldRef) -> Result<()> {
        self.depth -= 1;
        Ok(())
    }

    fn field(
        &mut self,
        field: &iceberg::spec::NestedFieldRef,
        value: String,
    ) -> iceberg::Result<String> {
        if self.is_inside_struct() {
            return Ok(format!("{}:{}", field.name, &value));
        }

        // Top-level columns are deduplicated by name across schema versions,
        // matching the Java (`IcebergToGlueConverter`) and PyIceberg
        // (`_to_columns`) implementations. The current schema is visited
        // first, so a name that appears in multiple schema versions keeps its
        // current type and `iceberg.field.current=true`. Without this,
        // every schema version re-emits the full column list and the Glue
        // `UpdateTable` payload grows until it exceeds Glue's request size
        // limit, permanently failing all commits on the table.
        if !self.seen_names.insert(field.name.clone()) {
            return Ok(value);
        }

        let parameters = HashMap::from([
            (ICEBERG_FIELD_ID.to_string(), format!("{}", field.id)),
            (
                ICEBERG_FIELD_OPTIONAL.to_string(),
                format!("{}", !field.required).to_lowercase(),
            ),
            (
                ICEBERG_FIELD_CURRENT.to_string(),
                format!("{}", self.is_current).to_lowercase(),
            ),
        ]);

        let mut builder = Column::builder()
            .name(field.name.clone())
            .r#type(&value)
            .set_parameters(Some(parameters));

        if let Some(comment) = field.doc.as_ref() {
            builder = builder.comment(comment);
        }

        let column = builder.build().map_err(from_aws_build_error)?;

        self.schema.push(column);

        Ok(value)
    }

    fn list(&mut self, _list: &iceberg::spec::ListType, value: String) -> iceberg::Result<String> {
        Ok(format!("array<{value}>"))
    }

    fn map(
        &mut self,
        _map: &iceberg::spec::MapType,
        key_value: String,
        value: String,
    ) -> iceberg::Result<String> {
        Ok(format!("map<{key_value},{value}>"))
    }

    fn primitive(&mut self, p: &iceberg::spec::PrimitiveType) -> iceberg::Result<Self::T> {
        let glue_type = match p {
            PrimitiveType::Boolean => "boolean".to_string(),
            PrimitiveType::Int => "int".to_string(),
            PrimitiveType::Long => "bigint".to_string(),
            PrimitiveType::Float => "float".to_string(),
            PrimitiveType::Double => "double".to_string(),
            PrimitiveType::Date => "date".to_string(),
            // Hive/Glue's `timestamp` and `timestamp_ns` types are
            // implicitly UTC, so tz-aware Iceberg timestamps map onto
            // them without semantic loss. Athena reads both as UTC.
            PrimitiveType::Timestamp | PrimitiveType::Timestamptz => "timestamp".to_string(),
            PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs => "timestamp_ns".to_string(),
            PrimitiveType::Time | PrimitiveType::String | PrimitiveType::Uuid => {
                "string".to_string()
            }
            PrimitiveType::Binary | PrimitiveType::Fixed(_) => "binary".to_string(),
            PrimitiveType::Decimal { precision, scale } => {
                format!("decimal({precision},{scale})")
            }
        };

        Ok(glue_type)
    }
}

#[cfg(test)]
mod tests {
    use iceberg::TableCreation;
    use iceberg::spec::{NestedField, Schema, TableMetadataBuilder};

    use super::*;

    fn create_metadata(schema: Schema) -> Result<TableMetadata> {
        let table_creation = TableCreation::builder()
            .name("my_table".to_string())
            .location("my_location".to_string())
            .schema(schema)
            .build();
        let metadata = TableMetadataBuilder::from_table_creation(table_creation)?
            .build()?
            .metadata;

        Ok(metadata)
    }

    /// Columns a single-schema table yields — the fresh-table case, where the
    /// current schema is the whole schema history.
    fn current_columns(schema: Schema) -> Result<Vec<Column>> {
        Ok(storage_descriptor_for_table(&create_metadata(schema)?)?
            .columns()
            .to_vec())
    }

    fn create_column(
        name: impl Into<String>,
        r#type: impl Into<String>,
        id: impl Into<String>,
        optional: bool,
    ) -> Result<Column> {
        let parameters = HashMap::from([
            (ICEBERG_FIELD_ID.to_string(), id.into()),
            (ICEBERG_FIELD_OPTIONAL.to_string(), optional.to_string()),
            (ICEBERG_FIELD_CURRENT.to_string(), "true".to_string()),
        ]);

        Column::builder()
            .name(name)
            .r#type(r#type)
            .set_comment(None)
            .set_parameters(Some(parameters))
            .build()
            .map_err(from_aws_build_error)
    }

    /// Every Iceberg primitive maps onto its Glue type string. The collapsing
    /// cases are the ones worth pinning: both tz-aware timestamps join their
    /// naive counterparts, and `time`/`uuid` degrade to `string`.
    #[test]
    fn test_primitive_type_mapping() -> Result<()> {
        let cases = [
            (PrimitiveType::Boolean, "boolean"),
            (PrimitiveType::Int, "int"),
            (PrimitiveType::Long, "bigint"),
            (PrimitiveType::Float, "float"),
            (PrimitiveType::Double, "double"),
            (
                PrimitiveType::Decimal {
                    precision: 12,
                    scale: 3,
                },
                "decimal(12,3)",
            ),
            (PrimitiveType::Date, "date"),
            (PrimitiveType::Time, "string"),
            (PrimitiveType::Timestamp, "timestamp"),
            (PrimitiveType::Timestamptz, "timestamp"),
            (PrimitiveType::TimestampNs, "timestamp_ns"),
            (PrimitiveType::TimestamptzNs, "timestamp_ns"),
            (PrimitiveType::String, "string"),
            (PrimitiveType::Uuid, "string"),
            (PrimitiveType::Fixed(8), "binary"),
            (PrimitiveType::Binary, "binary"),
        ];

        for (primitive, expected_type) in cases {
            // Declared id 42, expected id 1: table *creation* reassigns field
            // ids from FIRST_FIELD_ID (and rewrites the declared schema id),
            // so the declared value cannot survive. Only creation does this —
            // `register_table` and `update_table` pass through whatever ids
            // the existing metadata carries.
            let schema = Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(42, "c", primitive.into()).into(),
                ])
                .build()?;

            assert_eq!(current_columns(schema)?, [create_column(
                "c",
                expected_type,
                "1",
                false
            )?]);
        }

        Ok(())
    }

    /// A field's doc becomes the Glue column comment; an undocumented field
    /// carries none.
    #[test]
    fn test_field_doc_becomes_column_comment() -> Result<()> {
        let schema = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::optional(1, "documented", PrimitiveType::String.into())
                    .with_doc("what it holds")
                    .into(),
                NestedField::required(2, "undocumented", PrimitiveType::Long.into()).into(),
            ])
            .build()?;

        let columns = current_columns(schema)?;

        assert_eq!(columns[0].comment(), Some("what it holds"));
        assert_eq!(columns[1..], [create_column(
            "undocumented",
            "bigint",
            "2",
            false
        )?]);

        Ok(())
    }

    /// The contract downstream tooling relies on: what `GlueCatalog` writes
    /// into a table's `StorageDescriptor` is *exactly* what the public builder
    /// returns — whole descriptor, historical columns included. The two share
    /// one implementation, so this guards against a divergent descriptor being
    /// reintroduced on the catalog path.
    #[test]
    fn test_catalog_writes_exactly_the_public_descriptor() -> Result<()> {
        let historical = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::optional(9, "historical_only", PrimitiveType::String.into()).into(),
            ])
            .build()?;
        let current = Schema::builder()
            .with_schema_id(1)
            .with_fields(vec![
                NestedField::required(1, "id", PrimitiveType::Long.into()).into(),
                NestedField::optional(2, "label", PrimitiveType::String.into()).into(),
            ])
            .build()?;

        let metadata = create_metadata(historical)?
            .into_builder(None)
            .add_current_schema(current)?
            .build()?
            .metadata;
        let table = crate::utils::convert_to_glue_table(
            "table",
            "metadata".to_string(),
            &metadata,
            &HashMap::new(),
            None,
        )?;
        let written = table
            .storage_descriptor()
            .expect("glue table must carry a storage descriptor");

        // Wiring: the catalog descriptor IS the public one. Tautological while
        // the two share an implementation — it fires only if a divergent build
        // is reintroduced on the catalog path.
        assert_eq!(*written, storage_descriptor_for_table(&metadata)?);

        // Content, asserted independently of that shared implementation:
        // current schema first, historical-only column retained and flagged
        // non-current. A wrong three-column descriptor fails here.
        let observed: Vec<_> = written
            .columns()
            .iter()
            .map(|c| {
                (
                    c.name(),
                    c.r#type().expect("column has a type"),
                    c.parameters()
                        .and_then(|p| p.get(ICEBERG_FIELD_CURRENT))
                        .map(String::as_str)
                        .expect("column carries iceberg.field.current"),
                )
            })
            .collect();

        assert_eq!(observed, vec![
            ("id", "bigint", "true"),
            ("label", "string", "true"),
            ("historical_only", "string", "false"),
        ]);
        assert_eq!(written.location(), Some("my_location"));

        Ok(())
    }

    #[test]
    fn test_schema_with_simple_fields() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "c1",
                    "required": true,
                    "type": "boolean"
                },
                {
                    "id": 2,
                    "name": "c2",
                    "required": true,
                    "type": "int"
                },
                {
                    "id": 3,
                    "name": "c3",
                    "required": true,
                    "type": "long"
                },
                {
                    "id": 4,
                    "name": "c4",
                    "required": true,
                    "type": "float"
                },
                {
                    "id": 5,
                    "name": "c5",
                    "required": true,
                    "type": "double"
                },
                {
                    "id": 6,
                    "name": "c6",
                    "required": true,
                    "type": "decimal(2,2)"
                },
                {
                    "id": 7,
                    "name": "c7",
                    "required": true,
                    "type": "date"
                },
                {
                    "id": 8,
                    "name": "c8",
                    "required": true,
                    "type": "time"
                },
                {
                    "id": 9,
                    "name": "c9",
                    "required": true,
                    "type": "timestamp"
                },
                {
                    "id": 10,
                    "name": "c10",
                    "required": true,
                    "type": "string"
                },
                {
                    "id": 11,
                    "name": "c11",
                    "required": true,
                    "type": "uuid"
                },
                {
                    "id": 12,
                    "name": "c12",
                    "required": true,
                    "type": "fixed[4]"
                },
                {
                    "id": 13,
                    "name": "c13",
                    "required": true,
                    "type": "binary"
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![
            create_column("c1", "boolean", "1", false)?,
            create_column("c2", "int", "2", false)?,
            create_column("c3", "bigint", "3", false)?,
            create_column("c4", "float", "4", false)?,
            create_column("c5", "double", "5", false)?,
            create_column("c6", "decimal(2,2)", "6", false)?,
            create_column("c7", "date", "7", false)?,
            create_column("c8", "string", "8", false)?,
            create_column("c9", "timestamp", "9", false)?,
            create_column("c10", "string", "10", false)?,
            create_column("c11", "string", "11", false)?,
            create_column("c12", "binary", "12", false)?,
            create_column("c13", "binary", "13", false)?,
        ];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_structs() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "person",
                    "required": true,
                    "type": {
                        "type": "struct",
                        "fields": [
                            {
                                "id": 2,
                                "name": "name",
                                "required": true,
                                "type": "string"
                            },
                            {
                                "id": 3,
                                "name": "age",
                                "required": false,
                                "type": "int"
                            }
                        ]
                    }
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "person",
            "struct<name:string, age:int>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_struct_inside_list() -> Result<()> {
        let record = r#"
        {
            "schema-id": 1,
            "type": "struct",
            "fields": [
                {
                    "id": 1,
                    "name": "location",
                    "required": true,
                    "type": {
                        "type": "list",
                        "element-id": 2,
                        "element-required": true,
                        "element": {
                            "type": "struct",
                            "fields": [
                                {
                                    "id": 3,
                                    "name": "latitude",
                                    "required": false,
                                    "type": "float"
                                },
                                {
                                    "id": 4,
                                    "name": "longitude",
                                    "required": false,
                                    "type": "float"
                                }
                            ]
                        }
                    }
                }
            ]
        }
        "#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "location",
            "array<struct<latitude:float, longitude:float>>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_nested_maps() -> Result<()> {
        let record = r#"
            {
                "schema-id": 1,
                "type": "struct",
                "fields": [
                    {
                        "id": 1,
                        "name": "quux",
                        "required": true,
                        "type": {
                            "type": "map",
                            "key-id": 2,
                            "key": "string",
                            "value-id": 3,
                            "value-required": true,
                            "value": {
                                "type": "map",
                                "key-id": 4,
                                "key": "string",
                                "value-id": 5,
                                "value-required": true,
                                "value": "int"
                            }
                        }
                    }
                ]
            }
        "#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![create_column(
            "quux",
            "map<string,map<string,int>>",
            "1",
            false,
        )?];

        assert_eq!(result, expected);

        Ok(())
    }

    #[test]
    fn test_schema_with_optional_fields() -> Result<()> {
        let record = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "required_field",
                    "required": true,
                    "type": "string"
                },
                {
                    "id": 2,
                    "name": "optional_field",
                    "required": false,
                    "type": "int"
                }
            ]
        }"#;

        let schema = serde_json::from_str::<Schema>(record)?;
        let metadata = create_metadata(schema)?;

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let expected = vec![
            create_column("required_field", "string", "1", false)?,
            create_column("optional_field", "int", "2", true)?,
        ];

        assert_eq!(result, expected);
        Ok(())
    }

    /// Columns must be deduplicated by name across schema versions (matching
    /// Java's `IcebergToGlueConverter` and PyIceberg's `_to_columns`): the
    /// current schema's entry wins for repeated names, and historical-only
    /// names appear once with `iceberg.field.current=false`. Without dedup,
    /// every schema version re-emits the full column list and the Glue
    /// `UpdateTable` payload eventually exceeds the request size limit.
    #[test]
    fn test_multiple_schema_versions_deduplicate_columns() -> Result<()> {
        let v0 = r#"{
            "type": "struct",
            "schema-id": 0,
            "fields": [
                {
                    "id": 1,
                    "name": "kept",
                    "required": true,
                    "type": "int"
                },
                {
                    "id": 2,
                    "name": "dropped",
                    "required": false,
                    "type": "string"
                }
            ]
        }"#;
        let v1 = r#"{
            "type": "struct",
            "schema-id": 1,
            "fields": [
                {
                    "id": 1,
                    "name": "kept",
                    "required": true,
                    "type": "long"
                },
                {
                    "id": 3,
                    "name": "added",
                    "required": false,
                    "type": "string"
                }
            ]
        }"#;

        let schema_v0 = serde_json::from_str::<Schema>(v0)?;
        let schema_v1 = serde_json::from_str::<Schema>(v1)?;

        let metadata = create_metadata(schema_v0)?
            .into_builder(None)
            .add_current_schema(schema_v1)?
            .build()?
            .metadata;
        assert_eq!(metadata.schemas_iter().count(), 2);

        let result = GlueSchemaBuilder::from_iceberg(&metadata)?.build();

        let historical_column = |name: &str, r#type: &str, id: &str, optional: bool| {
            let parameters = HashMap::from([
                (ICEBERG_FIELD_ID.to_string(), id.to_string()),
                (ICEBERG_FIELD_OPTIONAL.to_string(), optional.to_string()),
                (ICEBERG_FIELD_CURRENT.to_string(), "false".to_string()),
            ]);
            Column::builder()
                .name(name)
                .r#type(r#type)
                .set_comment(None)
                .set_parameters(Some(parameters))
                .build()
                .map_err(from_aws_build_error)
        };

        // Current schema first ("kept" with its CURRENT type, then "added"),
        // then historical-only names ("dropped") — and "kept" exactly once.
        let expected = vec![
            create_column("kept", "bigint", "1", false)?,
            create_column("added", "string", "3", true)?,
            historical_column("dropped", "string", "2", true)?,
        ];

        assert_eq!(result, expected);
        Ok(())
    }
}
