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

//! The module contains the visitor for calculating NaN values in give arrow record batch.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, RecordBatch, StructArray};
use arrow_schema::DataType;

use crate::Result;
use crate::arrow::{ArrowArrayAccessor, FieldMatchMode};
use crate::spec::{
    ListType, MapType, NestedFieldRef, PrimitiveType, Schema, SchemaRef, SchemaWithPartnerVisitor,
    StructType, visit_struct_with_partner,
};

macro_rules! cast_and_update_cnt_map {
    ($t:ty, $col:ident, $self:ident, $field_id:ident) => {
        let nan_val_cnt = $col
            .as_any()
            .downcast_ref::<$t>()
            .unwrap()
            .iter()
            .filter(|value| value.map_or(false, |v| v.is_nan()))
            .count() as u64;

        match $self.nan_value_counts.entry($field_id) {
            Entry::Occupied(mut ele) => {
                let total_nan_val_cnt = ele.get() + nan_val_cnt;
                ele.insert(total_nan_val_cnt);
            }
            Entry::Vacant(v) => {
                v.insert(nan_val_cnt);
            }
        };
    };
}

macro_rules! count_float_nans {
    ($col:ident, $self:ident, $field_id:ident) => {
        match $col.data_type() {
            DataType::Float32 => {
                cast_and_update_cnt_map!(Float32Array, $col, $self, $field_id);
            }
            DataType::Float64 => {
                cast_and_update_cnt_map!(Float64Array, $col, $self, $field_id);
            }
            _ => {}
        }
    };
}

/// Visitor which counts and keeps track of NaN value counts in given record batch(s)
pub struct NanValueCountVisitor {
    /// Stores field ID to NaN value count mapping
    pub nan_value_counts: HashMap<i32, u64>,
    match_mode: FieldMatchMode,
}

impl SchemaWithPartnerVisitor<ArrayRef> for NanValueCountVisitor {
    type T = ();

    fn schema(
        &mut self,
        _schema: &Schema,
        _partner: &ArrayRef,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn field(
        &mut self,
        _field: &NestedFieldRef,
        _partner: &ArrayRef,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn r#struct(
        &mut self,
        _struct: &StructType,
        _partner: &ArrayRef,
        _results: Vec<Self::T>,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn list(&mut self, _list: &ListType, _list_arr: &ArrayRef, _value: Self::T) -> Result<Self::T> {
        Ok(())
    }

    fn map(
        &mut self,
        _map: &MapType,
        _partner: &ArrayRef,
        _key_value: Self::T,
        _value: Self::T,
    ) -> Result<Self::T> {
        Ok(())
    }

    fn primitive(&mut self, _p: &PrimitiveType, _col: &ArrayRef) -> Result<Self::T> {
        Ok(())
    }

    fn after_struct_field(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        let field_id = field.id;
        count_float_nans!(partner, self, field_id);
        Ok(())
    }

    fn after_list_element(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        let field_id = field.id;
        count_float_nans!(partner, self, field_id);
        Ok(())
    }

    fn after_map_key(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        let field_id = field.id;
        count_float_nans!(partner, self, field_id);
        Ok(())
    }

    fn after_map_value(&mut self, field: &NestedFieldRef, partner: &ArrayRef) -> Result<()> {
        let field_id = field.id;
        count_float_nans!(partner, self, field_id);
        Ok(())
    }
}

impl NanValueCountVisitor {
    /// Creates new instance of NanValueCountVisitor
    pub fn new() -> Self {
        Self::new_with_match_mode(FieldMatchMode::Id)
    }

    /// Creates new instance of NanValueCountVisitor with explicit match mode
    pub fn new_with_match_mode(match_mode: FieldMatchMode) -> Self {
        Self {
            nan_value_counts: HashMap::new(),
            match_mode,
        }
    }

    /// Compute nan value counts in given schema and record batch
    pub fn compute(&mut self, schema: SchemaRef, batch: RecordBatch) -> Result<()> {
        let arrow_arr_partner_accessor = ArrowArrayAccessor::new_with_match_mode(self.match_mode);

        let struct_arr = Arc::new(StructArray::from(batch)) as ArrayRef;
        visit_struct_with_partner(
            schema.as_struct(),
            &struct_arr,
            self,
            &arrow_arr_partner_accessor,
        )?;

        Ok(())
    }
}

impl Default for NanValueCountVisitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Instant;

    use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;

    use super::*;
    use crate::ErrorKind;
    use crate::arrow::schema_to_arrow_schema;
    use crate::spec::{NestedField, Type};

    const ROWS: usize = 64;

    fn arrow_field(id: i32, name: &str, data_type: DataType) -> Field {
        Field::new(name, data_type, true).with_metadata(HashMap::from([(
            PARQUET_FIELD_ID_META_KEY.to_string(),
            id.to_string(),
        )]))
    }

    /// Floats where every `period`-th row (offset by `id`) is NaN, so counts differ per field.
    fn floats(id: i32) -> impl Iterator<Item = Option<f64>> {
        let period = (id as usize % 7) + 2;
        (0..ROWS).map(move |row| match (row + id as usize) % period {
            0 => Some(f64::NAN),
            1 => None,
            _ => Some(row as f64),
        })
    }

    /// Leaf `i` of a struct: float64, float32 or int64, chosen by `i`.
    fn leaf(id: i32, i: usize) -> (NestedField, Field, ArrayRef) {
        let name = format!("c{id}");
        match i % 3 {
            0 => (
                NestedField::optional(id, &name, Type::Primitive(PrimitiveType::Double)),
                arrow_field(id, &name, DataType::Float64),
                Arc::new(floats(id).collect::<Float64Array>()),
            ),
            1 => (
                NestedField::optional(id, &name, Type::Primitive(PrimitiveType::Float)),
                arrow_field(id, &name, DataType::Float32),
                Arc::new(
                    floats(id)
                        .map(|v| v.map(|v| v as f32))
                        .collect::<Float32Array>(),
                ),
            ),
            _ => (
                NestedField::optional(id, &name, Type::Primitive(PrimitiveType::Long)),
                arrow_field(id, &name, DataType::Int64),
                Arc::new(Int64Array::from_iter((0..ROWS as i64).map(Some))),
            ),
        }
    }

    /// Moves every element to a new position, so arrow order never matches Iceberg order.
    fn shuffle<T>(items: Vec<T>) -> Vec<T> {
        let n = items.len();
        let mut slots: Vec<Option<T>> = items.into_iter().map(Some).collect();
        // 7 is coprime with every length used here, so this is a permutation.
        (0..n)
            .map(|i| slots[(i * 7 + 3) % n].take().unwrap())
            .collect()
    }

    /// Builds an Iceberg schema of `n` top-level fields, every tenth one a struct of five
    /// leaves, and a batch whose arrow columns (and struct children) come in another order.
    fn reordered_table(n: usize) -> (Schema, RecordBatch) {
        let mut next_id = 1;
        let mut take_id = || {
            next_id += 1;
            next_id - 1
        };
        let mut iceberg_fields = Vec::new();
        let mut columns: Vec<(FieldRef, ArrayRef)> = Vec::new();
        for i in 0..n {
            if i % 10 == 9 {
                let id = take_id();
                let children: Vec<_> = (0..5).map(|j| leaf(take_id(), j)).collect();
                let struct_type = StructType::new(
                    children
                        .iter()
                        .map(|(f, _, _)| Arc::new(f.clone()))
                        .collect(),
                );
                let arrow_children: Vec<(FieldRef, ArrayRef)> = shuffle(
                    children
                        .into_iter()
                        .map(|(_, f, a)| (Arc::new(f), a))
                        .collect(),
                );
                let array = StructArray::from(arrow_children);
                let name = format!("s{id}");
                iceberg_fields.push(Arc::new(NestedField::optional(
                    id,
                    &name,
                    Type::Struct(struct_type),
                )));
                columns.push((
                    Arc::new(arrow_field(id, &name, array.data_type().clone())),
                    Arc::new(array),
                ));
            } else {
                let (iceberg, arrow, array) = leaf(take_id(), i);
                iceberg_fields.push(Arc::new(iceberg));
                columns.push((Arc::new(arrow), array));
            }
        }
        let schema = Schema::builder()
            .with_fields(iceberg_fields)
            .build()
            .unwrap();
        let (fields, arrays): (Vec<_>, Vec<_>) = shuffle(columns).into_iter().unzip();
        let batch =
            RecordBatch::try_new(Arc::new(ArrowSchema::new(Fields::from(fields))), arrays).unwrap();
        (schema, batch)
    }

    /// Counts NaNs straight from the arrow columns and their field-id metadata.
    fn reference_counts(fields: &Fields, columns: &[ArrayRef], counts: &mut HashMap<i32, u64>) {
        for (field, column) in fields.iter().zip(columns) {
            let id: i32 = field.metadata()[PARQUET_FIELD_ID_META_KEY].parse().unwrap();
            let nans = match field.data_type() {
                DataType::Float64 => column
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .filter(|v| v.is_nan())
                    .count(),
                DataType::Float32 => column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .flatten()
                    .filter(|v| v.is_nan())
                    .count(),
                DataType::Struct(children) => {
                    let array = column.as_any().downcast_ref::<StructArray>().unwrap();
                    reference_counts(children, array.columns(), counts);
                    continue;
                }
                _ => continue,
            };
            *counts.entry(id).or_default() += nans as u64;
        }
    }

    #[test]
    fn test_nan_counts_with_reordered_columns_match_reference() {
        let (schema, batch) = reordered_table(300);
        let schema = Arc::new(schema);

        let mut expected = HashMap::new();
        for _ in 0..2 {
            reference_counts(batch.schema().fields(), batch.columns(), &mut expected);
        }
        assert!(expected.len() > 250, "test must cover many float fields");
        assert!(expected.values().any(|&n| n > 0));

        for mode in [FieldMatchMode::Id, FieldMatchMode::Name] {
            let mut visitor = NanValueCountVisitor::new_with_match_mode(mode);
            visitor.compute(schema.clone(), batch.clone()).unwrap();
            visitor.compute(schema.clone(), batch.clone()).unwrap();
            assert_eq!(visitor.nan_value_counts, expected, "match mode {mode:?}");
        }
    }

    #[test]
    fn test_nan_counts_missing_field_id_errors() {
        let (schema, batch) = reordered_table(30);
        let schema = Arc::new(schema);
        let nested_id = schema
            .as_struct()
            .fields()
            .iter()
            .find_map(|f| match f.field_type.as_ref() {
                Type::Struct(s) => Some(s.fields()[2].id),
                _ => None,
            })
            .unwrap();

        // Drop the field-id metadata of one top-level column and of one nested child.
        let strip = |field: &Field, id: i32| -> Field {
            if field.metadata()[PARQUET_FIELD_ID_META_KEY] == id.to_string() {
                field.clone().with_metadata(HashMap::new())
            } else {
                field.clone()
            }
        };
        for missing in [1, nested_id] {
            let fields: Vec<Field> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| {
                    let f = strip(f, missing);
                    match f.data_type() {
                        DataType::Struct(children) => {
                            let children: Fields =
                                children.iter().map(|c| strip(c, missing)).collect();
                            f.with_data_type(DataType::Struct(children))
                        }
                        _ => f,
                    }
                })
                .collect();
            let columns = batch
                .columns()
                .iter()
                .zip(&fields)
                .map(|(c, f)| match f.data_type() {
                    DataType::Struct(children) => {
                        let s = c.as_any().downcast_ref::<StructArray>().unwrap();
                        Arc::new(StructArray::new(
                            children.clone(),
                            s.columns().to_vec(),
                            None,
                        )) as ArrayRef
                    }
                    _ => c.clone(),
                })
                .collect();
            let stripped =
                RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).unwrap();

            let err = NanValueCountVisitor::new()
                .compute(schema.clone(), stripped)
                .unwrap_err();
            assert_eq!(err.kind(), ErrorKind::DataInvalid);
            assert_eq!(
                err.message(),
                format!("Field id {missing} not found in struct array")
            );
        }
    }

    /// Times `compute` on a wide batch like the Plan 86 compactor writes.
    /// Run with `cargo test --release -p iceberg --lib nan_counts_wide_batch_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn nan_counts_wide_batch_timing() {
        const COLUMNS: i32 = 2_731;
        const BATCH_ROWS: usize = 1_024;
        const RUNS: u32 = 20;

        let fields = (1..=COLUMNS).map(|id| {
            let ty = if id % 3 == 0 {
                PrimitiveType::Double
            } else {
                PrimitiveType::Long
            };
            Arc::new(NestedField::optional(
                id,
                format!("c{id}"),
                Type::Primitive(ty),
            ))
        });
        let schema = Arc::new(Schema::builder().with_fields(fields).build().unwrap());
        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let columns = arrow_schema
            .fields()
            .iter()
            .map(|f| match f.data_type() {
                DataType::Float64 => Arc::new(Float64Array::from_iter(
                    (0..BATCH_ROWS).map(|i| Some(if i % 5 == 0 { f64::NAN } else { i as f64 })),
                )) as ArrayRef,
                _ => Arc::new(Int64Array::from_iter((0..BATCH_ROWS as i64).map(Some))),
            })
            .collect();
        let batch = RecordBatch::try_new(arrow_schema, columns).unwrap();

        let mut visitor = NanValueCountVisitor::new();
        visitor.compute(schema.clone(), batch.clone()).unwrap();
        let start = Instant::now();
        for _ in 0..RUNS {
            visitor.compute(schema.clone(), batch.clone()).unwrap();
        }
        let per_batch = start.elapsed() / RUNS;
        println!(
            "compute: {COLUMNS} columns x {BATCH_ROWS} rows: {:.3} ms per batch",
            per_batch.as_secs_f64() * 1e3
        );
        assert_eq!(visitor.nan_value_counts[&3], (RUNS as u64 + 1) * 205);
    }
}
