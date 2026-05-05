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

use arrow_array::{ArrayRef, RecordBatch, StructArray, make_array};
use arrow_buffer::NullBuffer;
use arrow_schema::{DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

use crate::arrow::schema::schema_to_arrow_schema;
use crate::error::Result;
use crate::spec::Schema as IcebergSchema;
use crate::{Error, ErrorKind};

/// Help to project specific field from `RecordBatch`` according to the fields id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordBatchProjector {
    // A vector of vectors, where each inner vector represents the index path to access a specific field in a nested structure.
    // E.g. [[0], [1, 2]] means the first field is accessed directly from the first column,
    // while the second field is accessed from the second column and then from its third subcolumn (second column must be a struct column).
    field_indices: Vec<Vec<usize>>,
    // The Iceberg field IDs being projected, in projection order. Carried so
    // `project_record_batch` can locate columns by `PARQUET_FIELD_ID_META_KEY`
    // metadata at call time rather than relying solely on the construction-time
    // positional `field_indices` cache. Defends against top-level column
    // reordering between read and projection (e.g. DataFusion plan rewrites in
    // the compactor path between iceberg-rust's reader and the partition
    // splitter). Empty when constructed via `new` without explicit field IDs.
    target_field_ids: Vec<i64>,
    // The schema reference after projection. This schema is derived from the original schema based on the given field IDs.
    projected_schema: SchemaRef,
}

impl RecordBatchProjector {
    /// Init ArrowFieldProjector
    ///
    /// This function will iterate through the field and fetch the field from the original schema according to the field ids.
    /// The function to fetch the field id from the field is provided by `field_id_fetch_func`, return None if the field need to be skipped.
    /// This function will iterate through the nested fields if the field is a struct, `searchable_field_func` can be used to control whether
    /// iterate into the nested fields.
    pub(crate) fn new<F1, F2>(
        original_schema: SchemaRef,
        field_ids: &[i32],
        field_id_fetch_func: F1,
        searchable_field_func: F2,
    ) -> Result<Self>
    where
        F1: Fn(&Field) -> Result<Option<i64>>,
        F2: Fn(&Field) -> bool,
    {
        let mut field_indices = Vec::with_capacity(field_ids.len());
        let mut fields = Vec::with_capacity(field_ids.len());
        let mut target_field_ids = Vec::with_capacity(field_ids.len());
        for &id in field_ids {
            let mut field_index = vec![];
            let field = Self::fetch_field_index(
                original_schema.fields(),
                &mut field_index,
                id as i64,
                &field_id_fetch_func,
                &searchable_field_func,
            )?
            .ok_or_else(|| {
                Error::new(ErrorKind::Unexpected, "Field not found")
                    .with_context("field_id", id.to_string())
            })?;
            fields.push(field.clone());
            field_indices.push(field_index);
            target_field_ids.push(id as i64);
        }
        let delete_arrow_schema = Arc::new(Schema::new(fields));
        Ok(Self {
            field_indices,
            target_field_ids,
            projected_schema: delete_arrow_schema,
        })
    }

    /// Create RecordBatchProjector using Iceberg schema.
    ///
    /// This constructor converts the Iceberg schema to Arrow schema with field ID metadata,
    /// then uses the standard field ID lookup for projection.
    ///
    /// # Arguments
    /// * `iceberg_schema` - The Iceberg schema for field ID mapping  
    /// * `target_field_ids` - The field IDs to project
    pub fn from_iceberg_schema(
        iceberg_schema: Arc<IcebergSchema>,
        target_field_ids: &[i32],
    ) -> Result<Self> {
        let arrow_schema_with_ids = Arc::new(schema_to_arrow_schema(&iceberg_schema)?);

        let field_id_fetch_func = |field: &Field| -> Result<Option<i64>> {
            if let Some(value) = field.metadata().get(PARQUET_FIELD_ID_META_KEY) {
                let field_id = value.parse::<i32>().map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Failed to parse field id".to_string(),
                    )
                    .with_context("value", value)
                    .with_source(e)
                })?;
                Ok(Some(field_id as i64))
            } else {
                Ok(None)
            }
        };

        let searchable_field_func = |_field: &Field| -> bool { true };

        Self::new(
            arrow_schema_with_ids,
            target_field_ids,
            field_id_fetch_func,
            searchable_field_func,
        )
    }

    fn fetch_field_index<F1, F2>(
        fields: &Fields,
        index_vec: &mut Vec<usize>,
        target_field_id: i64,
        field_id_fetch_func: &F1,
        searchable_field_func: &F2,
    ) -> Result<Option<FieldRef>>
    where
        F1: Fn(&Field) -> Result<Option<i64>>,
        F2: Fn(&Field) -> bool,
    {
        for (pos, field) in fields.iter().enumerate() {
            let id = field_id_fetch_func(field)?;
            if let Some(id) = id
                && target_field_id == id
            {
                index_vec.push(pos);
                return Ok(Some(field.clone()));
            }
            if let DataType::Struct(inner) = field.data_type()
                && searchable_field_func(field)
                && let Some(res) = Self::fetch_field_index(
                    inner,
                    index_vec,
                    target_field_id,
                    field_id_fetch_func,
                    searchable_field_func,
                )?
            {
                index_vec.push(pos);
                return Ok(Some(res));
            }
        }
        Ok(None)
    }

    /// Return the reference of projected schema
    pub(crate) fn projected_schema_ref(&self) -> &SchemaRef {
        &self.projected_schema
    }

    /// Do projection with record batch.
    ///
    /// Prefers locating each target field by its `PARQUET_FIELD_ID_META_KEY`
    /// metadata in the input batch's schema (defends against top-level column
    /// reordering between construction and call — e.g. DataFusion plan
    /// rewrites in the compactor path). Falls back to the construction-time
    /// cached positional indices when the input batch lacks field-id metadata
    /// (back-compat for callers that build batches without it, and the only
    /// path that handles nested struct fields today).
    pub(crate) fn project_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        RecordBatch::try_new(
            self.projected_schema.clone(),
            self.project_record_batch(&batch)?,
        )
        .map_err(|err| Error::new(ErrorKind::DataInvalid, format!("{err}")))
    }

    /// Project columns from a `RecordBatch`, locating each target field by its
    /// `PARQUET_FIELD_ID_META_KEY` metadata in the batch schema when present,
    /// falling back to the construction-time cached positional indices when
    /// not. See [`project_batch`](Self::project_batch) for the rationale.
    pub fn project_record_batch(&self, batch: &RecordBatch) -> Result<Vec<ArrayRef>> {
        // Build a top-level field-id → column-index map from the input batch's
        // schema. Empty when no field carries the metadata key, in which case
        // every lookup falls back to the cached positional path.
        let id_to_top_level_idx = Self::build_top_level_field_id_index(batch.schema_ref())?;

        self.field_indices
            .iter()
            .enumerate()
            .map(|(projection_idx, cached_path)| {
                // Prefer metadata-driven lookup at the top level. Only the
                // top level is metadata-aware here; nested struct fields
                // continue to use cached positional traversal because the
                // existing nesting algorithm assumes positional stability
                // and no production caller exercises nested-virtual-partition
                // patterns today. Generalizing to nested metadata lookup is
                // tracked separately.
                let target_id = self.target_field_ids.get(projection_idx).copied();
                if cached_path.len() == 1
                    && let Some(id) = target_id
                {
                    if let Some(&top_level_idx) = id_to_top_level_idx.get(&id) {
                        return Self::get_column_by_field_index(batch.columns(), &[top_level_idx]);
                    }
                    // If the input batch carries field-id metadata for at
                    // least some columns but doesn't carry the target id,
                    // surface an explicit error rather than silently falling
                    // back to a positional index that would grab the wrong
                    // column. A batch with no metadata at all
                    // (id_to_top_level_idx empty) keeps the positional
                    // fallback for back-compat with non-iceberg callers.
                    if !id_to_top_level_idx.is_empty() {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            "Target field id not present in batch metadata",
                        )
                        .with_context("field_id", id.to_string()));
                    }
                }
                Self::get_column_by_field_index(batch.columns(), cached_path)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Do projection with columns (positional only; no metadata lookup).
    ///
    /// Retained for back-compat with callers that don't have a `RecordBatch`
    /// in scope. Prefer [`project_record_batch`](Self::project_record_batch)
    /// when a `RecordBatch` is available — it adds top-level field-id metadata
    /// resolution on top of the same positional path.
    pub fn project_column(&self, batch: &[ArrayRef]) -> Result<Vec<ArrayRef>> {
        self.field_indices
            .iter()
            .map(|index_vec| Self::get_column_by_field_index(batch, index_vec))
            .collect::<Result<Vec<_>>>()
    }

    /// Walk the top-level fields of `schema` and build a map from
    /// `PARQUET_FIELD_ID_META_KEY` value to column index. Fields without the
    /// metadata key are skipped. Returns an empty map when no field carries
    /// the metadata.
    fn build_top_level_field_id_index(
        schema: &SchemaRef,
    ) -> Result<std::collections::HashMap<i64, usize>> {
        let mut map = std::collections::HashMap::new();
        for (idx, field) in schema.fields().iter().enumerate() {
            if let Some(value) = field.metadata().get(PARQUET_FIELD_ID_META_KEY) {
                let id = value.parse::<i64>().map_err(|e| {
                    Error::new(ErrorKind::DataInvalid, "Failed to parse field id")
                        .with_context("value", value)
                        .with_source(e)
                })?;
                map.insert(id, idx);
            }
        }
        Ok(map)
    }

    fn get_column_by_field_index(batch: &[ArrayRef], field_index: &[usize]) -> Result<ArrayRef> {
        let mut rev_iterator = field_index.iter().rev();
        let mut array = batch[*rev_iterator.next().unwrap()].clone();
        let mut null_buffer = array.logical_nulls();
        for idx in rev_iterator {
            array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or(Error::new(
                    ErrorKind::Unexpected,
                    "Cannot convert Array to StructArray",
                ))?
                .column(*idx)
                .clone();
            null_buffer = NullBuffer::union(null_buffer.as_ref(), array.logical_nulls().as_ref());
        }
        Ok(make_array(
            array.to_data().into_builder().nulls(null_buffer).build()?,
        ))
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray, StructArray};
    use arrow_schema::{DataType, Field, Fields, Schema};

    use crate::arrow::record_batch_projector::RecordBatchProjector;
    use crate::spec::{NestedField, PrimitiveType, Schema as IcebergSchema, Type};
    use crate::{Error, ErrorKind};

    #[test]
    fn test_record_batch_projector_nested_level() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];
        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[1, 3], field_id_fetch_func, |_| true)
                .unwrap();

        assert_eq!(projector.field_indices.len(), 2);
        assert_eq!(projector.field_indices[0], vec![0]);
        assert_eq!(projector.field_indices[1], vec![0, 1]);

        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let inner_int_array = Arc::new(Int32Array::from(vec![4, 5, 6])) as ArrayRef;
        let inner_string_array = Arc::new(StringArray::from(vec!["x", "y", "z"])) as ArrayRef;
        let struct_array = Arc::new(StructArray::from(vec![
            (
                Arc::new(inner_fields[0].clone()),
                inner_int_array as ArrayRef,
            ),
            (
                Arc::new(inner_fields[1].clone()),
                inner_string_array as ArrayRef,
            ),
        ])) as ArrayRef;
        let batch = RecordBatch::try_new(schema, vec![int_array, struct_array]).unwrap();

        let projected_batch = projector.project_batch(batch).unwrap();
        assert_eq!(projected_batch.num_columns(), 2);
        let projected_int_array = projected_batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let projected_inner_int_array = projected_batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        assert_eq!(projected_int_array.values(), &[1, 2, 3]);
        assert_eq!(projected_inner_int_array.values(), &[4, 5, 6]);
    }

    #[test]
    fn test_field_not_found() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];

        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[1, 5], field_id_fetch_func, |_| true);

        assert!(projector.is_err());
    }

    #[test]
    fn test_field_not_reachable() {
        let inner_fields = vec![
            Field::new("inner_field1", DataType::Int32, false),
            Field::new("inner_field2", DataType::Utf8, false),
        ];

        let fields = vec![
            Field::new("field1", DataType::Int32, false),
            Field::new(
                "field2",
                DataType::Struct(Fields::from(inner_fields.clone())),
                false,
            ),
        ];
        let schema = Arc::new(Schema::new(fields));

        let field_id_fetch_func = |field: &Field| match field.name().as_str() {
            "field1" => Ok(Some(1)),
            "field2" => Ok(Some(2)),
            "inner_field1" => Ok(Some(3)),
            "inner_field2" => Ok(Some(4)),
            _ => Err(Error::new(ErrorKind::Unexpected, "Field id not found")),
        };
        let projector =
            RecordBatchProjector::new(schema.clone(), &[3], field_id_fetch_func, |_| false);
        assert!(projector.is_err());

        let projector =
            RecordBatchProjector::new(schema.clone(), &[3], field_id_fetch_func, |_| true);
        assert!(projector.is_ok());
    }

    #[test]
    fn test_from_iceberg_schema() {
        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let projector =
            RecordBatchProjector::from_iceberg_schema(Arc::new(iceberg_schema), &[1, 3]).unwrap();

        assert_eq!(projector.field_indices.len(), 2);
        assert_eq!(projector.projected_schema_ref().fields().len(), 2);
        assert_eq!(projector.projected_schema_ref().field(0).name(), "id");
        assert_eq!(projector.projected_schema_ref().field(1).name(), "age");
    }

    /// Regression for Fix 2 of plan-42 Task 7. When the input batch's columns
    /// are reordered relative to the construction-time iceberg schema (the
    /// shape produced by DataFusion plan rewrites in the compactor path),
    /// `project_record_batch` must locate columns by field-id metadata, not
    /// by cached positional index. Pre-Fix-2 this would silently grab the
    /// wrong column.
    #[test]
    fn test_project_record_batch_locates_columns_by_field_id_after_reorder() {
        use std::collections::HashMap;
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let projector =
            RecordBatchProjector::from_iceberg_schema(Arc::new(iceberg_schema), &[1, 3]).unwrap();

        // Construct an input batch where columns are in REVERSE iceberg
        // schema order (age, name, id), each carrying its iceberg field id
        // as parquet metadata. The cached positional indices say
        // [0]=field_id_1=id, [2]=field_id_3=age — applied positionally to
        // this reordered batch they would return age at slot 0 and ... an
        // out-of-bounds at slot 2 (only 3 cols). Field-id metadata lookup
        // must instead return id from idx 2 and age from idx 0.
        let with_id = |name: &str, dt: DataType, nullable: bool, id: i32| {
            Field::new(name, dt, nullable).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                id.to_string(),
            )]))
        };
        let arrow_schema = Arc::new(Schema::new(vec![
            with_id("age", DataType::Int32, true, 3),
            with_id("name", DataType::Utf8, false, 2),
            with_id("id", DataType::Int32, false, 1),
        ]));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![Some(40), Some(50), Some(60)])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        ])
        .unwrap();

        let projected = projector.project_record_batch(&batch).unwrap();
        assert_eq!(projected.len(), 2);

        let id_col = projected[0].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_col.values(), &[1, 2, 3]);

        let age_col = projected[1].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(age_col.value(0), 40);
        assert_eq!(age_col.value(1), 50);
        assert_eq!(age_col.value(2), 60);
    }

    /// Pin against silent grab-the-wrong-column when a metadata-tagged batch
    /// is missing a target field id entirely (partial projection). The
    /// projector must surface an explicit error instead of falling back to a
    /// positional index that could collide with an unrelated column.
    #[test]
    fn test_project_record_batch_errors_on_missing_field_id_in_tagged_batch() {
        use std::collections::HashMap;
        use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let projector =
            RecordBatchProjector::from_iceberg_schema(Arc::new(iceberg_schema), &[1, 3]).unwrap();

        // Batch has only fields 1 and 2 — field 3 (age) is absent. Metadata
        // is present, so we must NOT positional-fallback to slot [2] (which
        // would either OOB or grab a wrong column).
        let with_id = |name: &str, dt: DataType, nullable: bool, id: i32| {
            Field::new(name, dt, nullable).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                id.to_string(),
            )]))
        };
        let arrow_schema = Arc::new(Schema::new(vec![
            with_id("id", DataType::Int32, false, 1),
            with_id("name", DataType::Utf8, false, 2),
        ]));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ])
        .unwrap();

        let err = projector.project_record_batch(&batch).unwrap_err();
        assert!(
            err.to_string().contains("Target field id not present"),
            "expected explicit error about missing field id, got: {err}"
        );
    }

    /// Pin the back-compat positional fallback for batches that don't carry
    /// any field-id metadata. The projector falls back to the
    /// construction-time cached positional indices in this case (the path
    /// existing non-iceberg callers depend on).
    #[test]
    fn test_project_record_batch_falls_back_to_positional_when_no_metadata() {
        let iceberg_schema = IcebergSchema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();

        let projector =
            RecordBatchProjector::from_iceberg_schema(Arc::new(iceberg_schema), &[1, 3]).unwrap();

        // Batch in iceberg-schema order, NO field-id metadata. Cached
        // positional path should still work — id at slot 0, age at slot 2.
        let arrow_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("age", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
            Arc::new(Int32Array::from(vec![Some(40), Some(50), Some(60)])) as ArrayRef,
        ])
        .unwrap();

        let projected = projector.project_record_batch(&batch).unwrap();
        let id_col = projected[0].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(id_col.values(), &[1, 2, 3]);
        let age_col = projected[1].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(age_col.value(0), 40);
        assert_eq!(age_col.value(2), 60);
    }
}
