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

//! Local-only regression tests for commit validation and manifest allocation.

use std::collections::HashMap;

use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalog, MemoryCatalogBuilder};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, FormatVersion, Literal,
    NestedField, PrimitiveType, Schema, Struct, StructType, Type, read_data_files_from_avro,
    write_data_files_to_avro,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, CatalogBuilder, ErrorKind, NamespaceIdent, TableCreation};

fn file(path: String, content: DataContentType) -> DataFile {
    DataFileBuilder::default()
        .partition_spec_id(0)
        .content(content)
        .file_path(path)
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(100)
        .record_count(1)
        .partition(Struct::empty())
        .build()
        .unwrap()
}

fn path(manifest: usize, entry: usize) -> String {
    format!(
        "memory:///data/{manifest}/{entry}/{}.parquet",
        "x".repeat(512)
    )
}

async fn fixture(manifests: usize, entries: usize) -> (MemoryCatalog, Table) {
    let catalog = MemoryCatalogBuilder::default()
        .load(
            "test",
            HashMap::from([(
                MEMORY_CATALOG_WAREHOUSE.to_string(),
                "memory:///warehouse".to_string(),
            )]),
        )
        .await
        .unwrap();
    let namespace = NamespaceIdent::new("memory_hotfix".to_string());
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await
        .unwrap();
    let schema = Schema::builder()
        .with_fields([NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into()])
        .build()
        .unwrap();
    let mut table = catalog
        .create_table(
            &namespace,
            TableCreation::builder()
                .name("events".to_string())
                .schema(schema)
                .build(),
        )
        .await
        .unwrap();
    for index in 0..manifests {
        let tx = Transaction::new(&table);
        // Fixture paths are unique by construction. The measured append below
        // uses the default duplicate checks, exactly as the committer does.
        table = tx
            .fast_append()
            .set_check_duplicate(false)
            .add_data_files(
                (0..entries).map(|entry| file(path(index, entry), DataContentType::Data)),
            )
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
    }
    (catalog, table)
}

fn one_file_append_peak(manifests: usize, entries: usize) -> u64 {
    // All decode work runs on this thread; allocation-counter is thread-local.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (catalog, table) = runtime.block_on(fixture(manifests, entries));
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        result = Some(runtime.block_on(async {
            let tx = Transaction::new(&table);
            tx.fast_append()
                .add_data_files([file(
                    "memory:///data/new.parquet".to_string(),
                    DataContentType::Data,
                )])
                .apply(tx)
                .unwrap()
                .commit(&catalog)
                .await
        }));
    });
    let appended = result.unwrap().unwrap();
    assert_eq!(
        appended
            .metadata()
            .current_snapshot()
            .unwrap()
            .summary()
            .additional_properties["total-data-files"],
        (manifests * entries + 1).to_string(),
    );
    allocations.bytes_max
}

#[test]
fn one_file_append_does_not_retain_all_manifests() {
    let small = one_file_append_peak(4, 512);
    let large = one_file_append_peak(48, 512);
    eprintln!("one-file append peak allocated bytes: 4 manifests={small}; 48 manifests={large}");
    // 12x more equal-sized manifests must not cause a corresponding decoded
    // heap increase. Allow headroom for the larger manifest list and metadata.
    assert!(
        large < small * 3,
        "decoded inventory retained: small={small}, large={large}"
    );
}

#[test]
#[ignore = "2.5 million synthetic files; explicit local scale check"]
fn one_file_append_at_large_table_file_count() {
    let peak = one_file_append_peak(100, 25_000);
    eprintln!("one-file append / 2.5 million existing files: peak allocated bytes={peak}");
    assert!(
        peak < 8 * 1024 * 1024 * 1024,
        "synthetic append exceeds 8GiB: {peak}"
    );
}

#[test]
fn partition_data_files_preserve_values_and_nulls() {
    let partition_type = StructType::new(vec![
        NestedField::required(1000, "ingest_day", Type::Primitive(PrimitiveType::Int)).into(),
        NestedField::optional(1001, "product_name", Type::Primitive(PrimitiveType::String)).into(),
    ]);
    let schema = Schema::builder()
        .with_fields(partition_type.fields().to_vec())
        .build()
        .unwrap();
    let serialize = |present: bool| {
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path("memory:///partitioned.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter([
                Some(Literal::int(20000)),
                present.then(|| Literal::string("router")),
            ]))
            .build()
            .unwrap();
        let mut bytes = Vec::new();
        write_data_files_to_avro(
            &mut bytes,
            [data_file.clone()],
            &partition_type,
            FormatVersion::V2,
        )
        .unwrap();
        let decoded = read_data_files_from_avro(
            &mut bytes.as_slice(),
            &schema,
            0,
            &partition_type,
            FormatVersion::V2,
        )
        .unwrap();
        assert_eq!(decoded, vec![data_file]);
    };
    serialize(true);
    serialize(false);
}

#[tokio::test]
async fn late_manifest_duplicates_are_rejected_for_both_append_modes() {
    let (catalog, table) = fixture(4, 2).await;
    for merge in [false, true] {
        let tx = Transaction::new(&table);
        // The oldest file is in the final manifest, not the newly appended one.
        let duplicate = file(path(0, 0), DataContentType::Data);
        let tx = if merge {
            tx.merge_append()
                .add_data_files([duplicate])
                .apply(tx)
                .unwrap()
        } else {
            tx.fast_append()
                .add_data_files([duplicate])
                .apply(tx)
                .unwrap()
        };
        let error = tx.commit(&catalog).await.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DataInvalid);
        assert!(error.to_string().contains("already referenced"), "{error}");
    }
}

#[tokio::test]
async fn delete_file_duplicates_and_missing_removals_are_rejected() {
    let (catalog, table) = fixture(2, 2).await;
    let delete = file(
        "memory:///positions.parquet".to_string(),
        DataContentType::PositionDeletes,
    );
    let tx = Transaction::new(&table);
    let table = tx
        .fast_append()
        .add_data_files([delete.clone()])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    let error = tx
        .fast_append()
        .add_data_files([delete])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(error.to_string().contains("already referenced"));

    let tx = Transaction::new(&table);
    let error = tx
        .rewrite_files()
        .set_check_file_existence(true)
        .delete_files([file(
            "memory:///missing.parquet".to_string(),
            DataContentType::Data,
        )])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::DataInvalid);
    assert!(
        error.to_string().contains("not in the current snapshot"),
        "{error}"
    );
}

#[tokio::test]
async fn later_manifest_read_failure_is_not_hidden_after_finding_removal() {
    let (catalog, table) = fixture(2, 2).await;
    let list = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .load_manifest_list(table.file_io(), &table.metadata_ref())
        .await
        .unwrap();
    let first = list.entries()[0]
        .load_manifest(table.file_io())
        .await
        .unwrap();
    let removal = first.entries()[0].data_file().clone();
    let corrupt = &list.entries()[1].manifest_path;
    table
        .file_io()
        .new_output(corrupt)
        .unwrap()
        .write(bytes::Bytes::from_static(b"invalid avro"))
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    assert!(
        tx.rewrite_files()
            .set_check_file_existence(true)
            .delete_files([removal])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn deleted_entry_does_not_block_readding_a_path() {
    let (catalog, table) = fixture(2, 2).await;
    let removed = file(path(0, 0), DataContentType::Data);
    let tx = Transaction::new(&table);
    let table = tx
        .rewrite_files()
        .set_check_file_existence(true)
        .delete_files([removed.clone()])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
    let tx = Transaction::new(&table);
    tx.fast_append()
        .add_data_files([removed])
        .apply(tx)
        .unwrap()
        .commit(&catalog)
        .await
        .unwrap();
}
