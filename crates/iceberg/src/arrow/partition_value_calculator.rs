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

//! Partition value calculation for Iceberg tables.
//!
//! This module provides utilities for calculating partition values from record batches
//! based on a partition specification.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_cast::cast;
use arrow_schema::DataType;

use super::record_batch_projector::RecordBatchProjector;
use super::type_to_arrow_type;
use crate::spec::{PartitionSpec, Schema, StructType, Type};
use crate::transform::{BoxedTransformFunction, create_transform_function};
use crate::{Error, ErrorKind, Result};

/// Calculator for partition values in Iceberg tables.
///
/// This struct handles the projection of source columns and application of
/// partition transforms to compute partition values for a given record batch.
#[derive(Debug)]
pub struct PartitionValueCalculator {
    projector: RecordBatchProjector,
    transform_functions: Vec<BoxedTransformFunction>,
    partition_type: StructType,
    partition_arrow_type: DataType,
}

impl PartitionValueCalculator {
    /// Create a new PartitionValueCalculator.
    ///
    /// # Arguments
    ///
    /// * `partition_spec` - The partition specification
    /// * `table_schema` - The Iceberg table schema
    ///
    /// # Returns
    ///
    /// Returns a new `PartitionValueCalculator` instance or an error if initialization fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The partition spec is unpartitioned
    /// - Transform function creation fails
    /// - Projector initialization fails
    pub fn try_new(partition_spec: &PartitionSpec, table_schema: &Schema) -> Result<Self> {
        if partition_spec.is_unpartitioned() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot create partition calculator for unpartitioned table",
            ));
        }

        // Create transform functions for each partition field
        let transform_functions: Vec<BoxedTransformFunction> = partition_spec
            .fields()
            .iter()
            .map(|pf| create_transform_function(&pf.transform))
            .collect::<Result<Vec<_>>>()?;

        // Extract source field IDs for projection
        let source_field_ids: Vec<i32> = partition_spec
            .fields()
            .iter()
            .map(|pf| pf.source_id)
            .collect();

        // Create projector for extracting source columns
        let projector = RecordBatchProjector::from_iceberg_schema(
            Arc::new(table_schema.clone()),
            &source_field_ids,
        )?;

        // Get partition type information
        let partition_type = partition_spec.partition_type(table_schema)?;
        let partition_arrow_type = type_to_arrow_type(&Type::Struct(partition_type.clone()))?;

        Ok(Self {
            projector,
            transform_functions,
            partition_type,
            partition_arrow_type,
        })
    }

    /// Get the partition type as an Iceberg StructType.
    pub fn partition_type(&self) -> &StructType {
        &self.partition_type
    }

    /// Get the partition type as an Arrow DataType.
    pub fn partition_arrow_type(&self) -> &DataType {
        &self.partition_arrow_type
    }

    /// Calculate partition values for a record batch.
    ///
    /// This method:
    /// 1. Projects the source columns from the batch
    /// 2. Applies partition transforms to each source column
    /// 3. Constructs a StructArray containing the partition values
    ///
    /// # Arguments
    ///
    /// * `batch` - The record batch to calculate partition values for
    ///
    /// # Returns
    ///
    /// Returns an ArrayRef containing a StructArray of partition values, or an error if calculation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Column projection fails
    /// - Transform application fails
    /// - StructArray construction fails
    pub fn calculate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        // Project source columns from the batch. Prefer metadata-driven
        // lookup (`PARQUET_FIELD_ID_META_KEY`) over the projector's cached
        // positional indices — defends against top-level column reordering
        // between the iceberg-rust reader and the splitter (e.g. DataFusion
        // plan rewrites in the compactor path). Falls back to positional
        // indexing when the input batch lacks field-id metadata.
        let source_columns = self.projector.project_record_batch(batch)?;

        // Get expected struct fields for the result
        let expected_struct_fields = match &self.partition_arrow_type {
            DataType::Struct(fields) => fields.clone(),
            _ => {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Expected partition type must be a struct",
                ));
            }
        };

        // Apply transforms to each source column, then normalize the result
        // to the expected partition arrow type. Identity transform is a
        // pass-through, so when the read side delivers a partition source
        // column as a non-flat encoding (e.g. `RunEndEncoded` constant from
        // `RecordBatchTransformer::insert_constant_field`, or `Dictionary`
        // from a DataFusion plan), the transformed value carries that
        // encoding through. `StructArray::try_new` requires each child to
        // match its declared field type exactly, so we must decode to the
        // canonical flat encoding before constructing the struct.
        //
        // `arrow_cast::cast` is the canonical primitive: passthrough for
        // already-matching types, decode for REE/Dictionary → flat, loud
        // failure (Result) for genuinely incompatible types. No silent
        // NULL coercion.
        let mut partition_values = Vec::with_capacity(self.transform_functions.len());
        for ((source_column, transform_fn), expected_field) in source_columns
            .iter()
            .zip(&self.transform_functions)
            .zip(expected_struct_fields.iter())
        {
            let transformed = transform_fn.transform(source_column.clone())?;
            let normalized = if transformed.data_type() == expected_field.data_type() {
                transformed
            } else {
                cast(&transformed, expected_field.data_type()).map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Failed to cast partition value for field '{}' from {:?} to {:?}: {e}",
                            expected_field.name(),
                            transformed.data_type(),
                            expected_field.data_type(),
                        ),
                    )
                })?
            };
            partition_values.push(normalized);
        }

        // Construct the StructArray
        let struct_array = StructArray::try_new(expected_struct_fields, partition_values, None)
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to create partition struct array: {e}"),
                )
            })?;

        Ok(Arc::new(struct_array))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, DictionaryArray, Int32Array, RecordBatch, RunArray, StringArray};
    use arrow_array::types::Int32Type;
    use arrow_schema::{Field, Schema as ArrowSchema};

    use super::*;
    use crate::spec::{NestedField, PartitionSpecBuilder, PrimitiveType, Transform};

    #[test]
    fn test_partition_calculator_identity_transform() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .add_partition_field("id", "id_partition", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        // Verify partition type
        assert_eq!(calculator.partition_type().fields().len(), 1);
        assert_eq!(calculator.partition_type().fields()[0].name, "id_partition");

        // Create test batch
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ])
        .unwrap();

        // Calculate partition values
        let result = calculator.calculate(&batch).unwrap();
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();

        let id_partition = struct_array
            .column_by_name("id_partition")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(id_partition.value(0), 10);
        assert_eq!(id_partition.value(1), 20);
        assert_eq!(id_partition.value(2), 30);
    }

    /// Reproduces F3 from plan-42 task 6: when the read-side fix injects an
    /// identity-partition virtual column as a `RunEndEncoded` constant array
    /// (the shape produced by `RecordBatchTransformer::insert_constant_field`),
    /// the calculator must still yield the constant value through to the
    /// partition struct. Pre-fix this returned a NULL partition value and the
    /// downstream rewriter wrote files at `<col>=null/` paths.
    #[test]
    fn test_partition_calculator_identity_on_run_end_encoded() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "product_name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .add_partition_field("product_name", "product_name", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        // Build a RecordBatch where `product_name` is a RunEndEncoded constant
        // (one run, three logical rows, value "cisco_meraki_events"), matching
        // the shape `RecordBatchTransformer` produces for injected partition
        // constants.
        let run_ends = Int32Array::from(vec![3]);
        let values = StringArray::from(vec!["cisco_meraki_events"]);
        let ree_array: ArrayRef =
            Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("product_name", ree_array.data_type().clone(), true),
        ]));

        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            ree_array,
        ])
        .unwrap();

        let result = calculator.calculate(&batch).unwrap();
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();

        // Per Iceberg spec: identity-partitioned virtual columns must surface
        // their constant value into the partition struct, regardless of whether
        // the source column is encoded as Plain, Dictionary, or RunEndEncoded.
        let product_partition = struct_array
            .column_by_name("product_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("partition value column should be a plain StringArray");
        assert_eq!(product_partition.len(), 3);
        assert!(!product_partition.is_null(0), "row 0 partition value is NULL — read-side REE constant did not flow through");
        assert_eq!(product_partition.value(0), "cisco_meraki_events");
        assert_eq!(product_partition.value(1), "cisco_meraki_events");
        assert_eq!(product_partition.value(2), "cisco_meraki_events");
    }

    /// Sibling regression for the dictionary-encoded source column path.
    /// DataFusion can hand the splitter a `Dictionary<Int32, Utf8>` partition
    /// source column (cast/group-by intermediates can produce these). The
    /// calculator must decode to the canonical flat encoding before
    /// `StructArray::try_new`.
    #[test]
    fn test_partition_calculator_identity_on_dictionary_encoded() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "product_name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .add_partition_field("product_name", "product_name", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        let keys = Int32Array::from(vec![0, 0, 0]);
        let values = StringArray::from(vec!["cisco_meraki_events"]);
        let dict_array: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap(),
        );

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("product_name", dict_array.data_type().clone(), true),
        ]));

        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            dict_array,
        ])
        .unwrap();

        let result = calculator.calculate(&batch).unwrap();
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();
        let product_partition = struct_array
            .column_by_name("product_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("partition value column should be a plain StringArray");
        assert_eq!(product_partition.len(), 3);
        assert_eq!(product_partition.value(0), "cisco_meraki_events");
        assert_eq!(product_partition.value(1), "cisco_meraki_events");
        assert_eq!(product_partition.value(2), "cisco_meraki_events");
    }

    /// Pin against accidentally promoting genuine NULLs to inferred constants.
    /// Iceberg spec column-projection rule #2 says fields missing from the data
    /// file resolve to NULL (only rule #1 — the manifest fallback for identity
    /// partitions — substitutes a value). The calculator must surface NULL
    /// when the source column is legitimately NULL, not coerce it.
    #[test]
    fn test_partition_calculator_null_source_remains_null() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::optional(2, "product_name", Type::Primitive(PrimitiveType::String))
                    .into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .add_partition_field("product_name", "product_name", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        let calculator = PartitionValueCalculator::try_new(&partition_spec, &table_schema).unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("product_name", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec![
                Option::<&str>::None,
                Option::<&str>::None,
                Option::<&str>::None,
            ])),
        ])
        .unwrap();

        let result = calculator.calculate(&batch).unwrap();
        let struct_array = result.as_any().downcast_ref::<StructArray>().unwrap();
        let product_partition = struct_array
            .column_by_name("product_name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(product_partition.len(), 3);
        assert!(product_partition.is_null(0));
        assert!(product_partition.is_null(1));
        assert!(product_partition.is_null(2));
    }

    #[test]
    fn test_partition_calculator_unpartitioned_error() {
        let table_schema = Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let partition_spec = PartitionSpecBuilder::new(Arc::new(table_schema.clone()))
            .build()
            .unwrap();

        let result = PartitionValueCalculator::try_new(&partition_spec, &table_schema);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("unpartitioned table")
        );
    }
}
