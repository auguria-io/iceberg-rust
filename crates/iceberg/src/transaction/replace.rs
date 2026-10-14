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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::snapshot::{DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer};
use super::{ActionCommit, Transaction, TransactionAction};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestEntry,
    ManifestFile, Operation,
};
use crate::table::Table;
use crate::{Error, ErrorKind, Result};

impl Transaction {
    /// Replaces exactly the given data files with the given outputs, reading
    /// only the manifests that hold the inputs.
    ///
    /// Each removed input names the base manifest that holds it. The action
    /// refuses, before writing anything, unless every input is live in its
    /// named manifest with an equal descriptor. Untouched manifests are
    /// carried by reference. A touched manifest is rewritten with its other
    /// live entries kept as `Existing`, sequence numbers unchanged, and
    /// without its older tombstones. The inputs are recorded as `Deleted`
    /// under the new snapshot. Format V2 data files only; delete files in
    /// the inputs, outputs or base are refused, and so is a partitioned input
    /// under a spec other than the table's default.
    ///
    /// The bound covers reads only: inputs live elsewhere, and output paths
    /// live in untouched manifests, are not detected here.
    pub fn prepare_replace(
        mut self,
        removed: impl IntoIterator<Item = (String, DataFile)>,
        added: impl IntoIterator<Item = DataFile>,
    ) -> Self {
        self.actions.push(Arc::new(ReplaceAction {
            removed: removed.into_iter().collect(),
            added: added.into_iter().collect(),
        }));
        self
    }
}

struct ReplaceAction {
    /// Removed inputs, each with the path of the base manifest holding it.
    removed: Vec<(String, DataFile)>,
    added: Vec<DataFile>,
}

impl ReplaceAction {
    /// Checks the lists without reading the table.
    fn check_lists(&self) -> Result<()> {
        if self.removed.is_empty() || self.added.is_empty() {
            return Err(invalid(
                "Replace needs at least one removed and one added file",
            ));
        }
        let files = self.removed.iter().map(|(_, file)| file).chain(&self.added);
        if let Some(file) = files
            .clone()
            .find(|f| f.content_type() != DataContentType::Data)
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Replace supports data files only: {}", file.file_path()),
            ));
        }
        let mut paths = HashSet::new();
        if let Some(file) = files.into_iter().find(|f| !paths.insert(f.file_path())) {
            return Err(invalid(format!(
                "Replace lists file {} more than once",
                file.file_path()
            )));
        }
        Ok(())
    }

    /// Reads the manifest list and the touched manifests, and checks every
    /// input against them.
    async fn plan(&self, table: &Table) -> Result<ReplaceOperation> {
        if table.metadata().format_version() != FormatVersion::V2 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "Replace supports format version 2 tables only",
            ));
        }
        // The snapshot summary reads every removed file's partition through the
        // default spec; a partitioned file under another spec would not fit it.
        let default_spec_id = table.metadata().default_partition_spec_id();
        let summarisable = |f: &DataFile| {
            f.partition_spec_id == default_spec_id || f.partition().fields().is_empty()
        };
        if let Some(file) = self
            .removed
            .iter()
            .map(|(_, f)| f)
            .find(|f| !summarisable(f))
        {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Replace input {} is partitioned under spec {}, not the default spec {default_spec_id}",
                    file.file_path(),
                    file.partition_spec_id
                ),
            ));
        }
        let snapshot = table
            .metadata()
            .snapshot_for_ref(MAIN_BRANCH)
            .ok_or_else(|| stale("Replace base has no current snapshot"))?;
        let manifests = snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await?;

        let mut inputs: HashMap<&str, HashMap<&str, &DataFile>> = HashMap::new();
        for (manifest, file) in &self.removed {
            inputs
                .entry(manifest.as_str())
                .or_default()
                .insert(file.file_path(), file);
        }
        let added: HashSet<&str> = self.added.iter().map(|f| f.file_path()).collect();
        let removed: HashSet<&str> = self.removed.iter().map(|(_, f)| f.file_path()).collect();

        if let Some(manifest) = manifests.entries().iter().find(|m| {
            m.content == ManifestContentType::Deletes
                && (m.has_added_files() || m.has_existing_files())
        }) {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "Replace does not support tables with delete files: {}",
                    manifest.manifest_path
                ),
            ));
        }

        let mut planned = Vec::new();
        for manifest in manifests.entries() {
            let Some(mut wanted) = inputs.remove(manifest.manifest_path.as_str()) else {
                planned.push(Planned::Carried(manifest.clone()));
                continue;
            };
            if manifest.content != ManifestContentType::Data {
                return Err(stale(format!(
                    "Replace input manifest {} is not a data manifest",
                    manifest.manifest_path
                )));
            }

            let (entries, _) = manifest.load_manifest(table.file_io()).await?.into_parts();
            let mut kept = Vec::new();
            let mut deleted = Vec::new();
            // Older tombstones are dropped, never copied.
            for entry in entries.into_iter().filter(|e| e.is_alive()) {
                let path = entry.file_path();
                if added.contains(path) {
                    return Err(invalid(format!("Replace output {path} is already live")));
                }
                match wanted.remove(path) {
                    Some(input) if input == entry.data_file() => {
                        deleted.push(Arc::unwrap_or_clone(entry))
                    }
                    Some(_) => {
                        return Err(stale(format!(
                            "Replace input {path} differs from its live descriptor"
                        )));
                    }
                    None if removed.contains(path) => {
                        return Err(invalid(format!("Replace input {path} is live twice")));
                    }
                    None => kept.push(Arc::unwrap_or_clone(entry)),
                }
            }
            if let Some(path) = wanted.into_keys().next() {
                return Err(stale(format!(
                    "Replace input {path} is not live in manifest {}",
                    manifest.manifest_path
                )));
            }
            planned.push(Planned::Rewritten {
                spec_id: manifest.partition_spec_id,
                kept,
                deleted,
            });
        }
        if let Some(manifest) = inputs.into_keys().next() {
            return Err(stale(format!(
                "Replace input manifest {manifest} is not in the base snapshot"
            )));
        }

        Ok(ReplaceOperation(planned))
    }
}

#[async_trait]
impl TransactionAction for ReplaceAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        self.check_lists()?;
        let operation = self.plan(table).await?;
        let removed = self.removed.iter().map(|(_, file)| file.clone()).collect();
        let producer = SnapshotProducer::new(
            table,
            Uuid::now_v7(),
            None,
            None,
            HashMap::new(),
            self.added.clone(),
            Vec::new(),
            removed,
            Vec::new(),
        );
        producer.validate_added_data_files(&self.added)?;

        // No validation pass and no manifest merge: both read every manifest.
        producer.commit(operation, DefaultManifestProcess).await
    }
}

/// The new snapshot's existing manifests, in base order.
struct ReplaceOperation(Vec<Planned>);

enum Planned {
    /// An untouched base manifest, carried by reference.
    Carried(ManifestFile),
    /// A touched base manifest: its other live entries and its removed inputs.
    Rewritten {
        spec_id: i32,
        kept: Vec<ManifestEntry>,
        deleted: Vec<ManifestEntry>,
    },
}

impl SnapshotProduceOperation for ReplaceOperation {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    /// The removed inputs are recorded in their rewritten manifests instead.
    async fn delete_entries(&self, _: &SnapshotProducer<'_>) -> Result<Vec<ManifestEntry>> {
        Ok(Vec::new())
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &mut SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let mut manifests = Vec::with_capacity(self.0.len());
        for planned in &self.0 {
            match planned {
                Planned::Carried(manifest) => manifests.push(manifest.clone()),
                Planned::Rewritten {
                    spec_id,
                    kept,
                    deleted,
                } => {
                    let mut writer = snapshot_produce
                        .new_manifest_writer(ManifestContentType::Data, *spec_id)?;
                    for entry in kept {
                        writer.add_existing_entry(entry.clone())?;
                    }
                    for entry in deleted {
                        writer.add_delete_entry(entry.clone())?;
                    }
                    manifests.push(writer.write_manifest_file().await?);
                }
            }
        }
        Ok(manifests)
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

fn stale(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::PreconditionFailed, message)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};
    use std::sync::{Arc, Mutex};

    use futures::TryStreamExt;

    use super::*;
    use crate::io::{FileIO, IoLog};
    use crate::spec::ManifestStatus::{self, Added, Deleted, Existing};
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Literal, ManifestFile,
        NestedField, PrimitiveType, Schema, Struct, TableMetadataBuilder, Transform, Type,
        UnboundPartitionSpec,
    };
    use crate::transaction::ApplyTransactionAction;
    use crate::{Catalog, MetadataLocation, TableCommit, TableCreation, TableIdent, TableUpdate};

    const S1: i64 = 1001;
    const S2: i64 = 1002;
    const S3: i64 = 1003;
    const S4: i64 = 1004;
    const S5: i64 = 1005;

    fn file(name: &str, rows: u64, content: DataContentType) -> DataFile {
        DataFileBuilder::default()
            .partition_spec_id(0)
            .content(content)
            .file_path(format!("memory:///t/data/{name}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(rows)
            .partition(Struct::empty())
            .build()
            .unwrap()
    }

    fn data(name: &str, rows: u64) -> DataFile {
        file(name, rows, DataContentType::Data)
    }

    fn name(path: &str) -> String {
        path.trim_start_matches("memory:///t/data/")
            .trim_end_matches(".parquet")
            .to_string()
    }

    /// A manifest entry: file name, status, snapshot id, data and file sequence.
    type Entry = (String, ManifestStatus, i64, i64, i64);

    fn entry(name: &str, status: ManifestStatus, snapshot: i64, seq: i64, file_seq: i64) -> Entry {
        (name.to_string(), status, snapshot, seq, file_seq)
    }

    fn empty_table(file_io: FileIO, format_version: FormatVersion) -> Table {
        let schema = Schema::builder()
            .with_fields([
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap();
        let metadata = TableMetadataBuilder::from_table_creation(
            TableCreation::builder()
                .name("t".to_string())
                .location("memory:///t".to_string())
                .schema(schema)
                .format_version(format_version)
                .build(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata;
        Table::builder()
            .file_io(file_io)
            .metadata(metadata)
            .metadata_location(MetadataLocation::new_with_table_location("memory:///t").to_string())
            .identifier(TableIdent::from_strs(["db", "t"]).unwrap())
            .build()
            .unwrap()
    }

    /// Applies a transaction the way a catalog does, without a catalog.
    async fn commit(table: Table, tx: Transaction) -> Result<Table> {
        let (updates, requirements) = tx.apply_actions().await?;
        TableCommit::builder()
            .ident(table.identifier().clone())
            .updates(updates)
            .requirements(requirements)
            .build()
            .apply(table)
    }

    async fn append(table: Table, snapshot_id: i64, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .set_snapshot_id(snapshot_id)
            .add_data_files(files)
            .apply(tx)
            .unwrap();
        commit(table, tx).await.unwrap()
    }

    async fn manifest_files(table: &Table) -> Vec<ManifestFile> {
        table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap()
            .entries()
            .to_vec()
    }

    async fn entries(table: &Table, manifest: &ManifestFile) -> Vec<Entry> {
        let mut entries: Vec<_> = manifest
            .load_manifest(table.file_io())
            .await
            .unwrap()
            .entries()
            .iter()
            .map(|e| {
                entry(
                    &name(e.file_path()),
                    e.status(),
                    e.snapshot_id().unwrap(),
                    e.sequence_number().unwrap(),
                    e.file_sequence_number().unwrap(),
                )
            })
            .collect();
        entries.sort_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));
        entries
    }

    async fn live_files(table: &Table) -> BTreeSet<String> {
        table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .map_ok(|task| name(task.data_file_path()))
            .try_collect()
            .await
            .unwrap()
    }

    /// S1 appends a and b; S2 appends c; S3 is a merged rewrite of c into d,
    /// leaving one mixed manifest Mx = {a, b Existing; d Added; c Deleted};
    /// S4 appends f (Mf); S5 appends g (Mg).
    struct Fixture {
        base: Table,
        log: Arc<Mutex<IoLog>>,
        mx: String,
        mf: String,
        mg: String,
    }

    impl Fixture {
        async fn new() -> Self {
            let (file_io, log) = FileIO::memory_with_log();
            let table = empty_table(file_io, FormatVersion::V2);
            let table = append(table, S1, vec![data("a", 1), data("b", 2)]).await;
            let table = append(table, S2, vec![data("c", 4)]).await;
            let tx = Transaction::new(&table);
            let mut rewrite = tx
                .rewrite_files()
                .set_snapshot_id(S3)
                .delete_files([data("c", 4)])
                .add_data_files([data("d", 8)]);
            rewrite.set_snapshot_properties(HashMap::from([
                (
                    "commit.manifest-merge.enabled".to_string(),
                    "true".to_string(),
                ),
                (
                    "commit.manifest.min-count-to-merge".to_string(),
                    "2".to_string(),
                ),
            ]));
            let table = commit(table, rewrite.apply(tx).unwrap()).await.unwrap();
            let table = append(table, S4, vec![data("f", 16)]).await;
            let table = append(table, S5, vec![data("g", 32)]).await;

            let paths: Vec<_> = manifest_files(&table)
                .await
                .into_iter()
                .map(|m| m.manifest_path)
                .collect();
            let [mx, mf, mg] = paths.try_into().unwrap();
            let fixture = Self {
                base: table,
                log,
                mx,
                mf,
                mg,
            };
            assert_eq!(
                entries(&fixture.base, &fixture.manifest(&fixture.mx).await).await,
                vec![
                    entry("a", Existing, S1, 1, 1),
                    entry("b", Existing, S1, 1, 1),
                    entry("c", Deleted, S3, 2, 2),
                    entry("d", Added, S3, 3, 3),
                ],
                "fixture intent"
            );
            fixture
        }

        async fn manifest(&self, path: &str) -> ManifestFile {
            manifest_files(&self.base)
                .await
                .into_iter()
                .find(|m| m.manifest_path == path)
                .unwrap()
        }

        /// Runs `prepare_replace` on the base, with an empty IO log.
        async fn replace(
            &self,
            removed: Vec<(&str, DataFile)>,
            added: Vec<DataFile>,
        ) -> Result<Table> {
            *self.log.lock().unwrap() = IoLog::default();
            let removed = removed
                .into_iter()
                .map(|(manifest, file)| (manifest.to_string(), file));
            let tx = Transaction::new(&self.base).prepare_replace(removed, added);
            commit(self.base.clone(), tx).await
        }

        /// Removes a (in Mx) and f (in Mf), adds h.
        async fn standard(&self) -> Table {
            self.replace(
                vec![(&self.mx, data("a", 1)), (&self.mf, data("f", 16))],
                vec![data("h", 64)],
            )
            .await
            .unwrap()
        }

        fn writes(&self) -> Vec<String> {
            self.log.lock().unwrap().writes.clone()
        }
    }

    async fn assert_refused(fixture: &Fixture, result: Result<Table>, kind: ErrorKind) {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), kind, "{error}");
        assert_eq!(
            fixture.writes(),
            Vec::<String>::new(),
            "refusal wrote files"
        );
    }

    #[tokio::test]
    async fn prepare_equivalence() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;
        let snapshot = table.metadata().current_snapshot().unwrap();
        let s6 = snapshot.snapshot_id();
        assert_eq!(snapshot.sequence_number(), 6);
        assert_eq!(snapshot.parent_snapshot_id(), Some(S5));

        let manifests = manifest_files(&table).await;
        let mut got = Vec::new();
        for manifest in &manifests {
            got.push(entries(&table, manifest).await);
        }
        assert_eq!(got, vec![
            vec![
                entry("a", Deleted, s6, 1, 1),
                entry("b", Existing, S1, 1, 1),
                entry("d", Existing, S3, 3, 3),
            ],
            vec![entry("f", Deleted, s6, 4, 4)],
            vec![entry("g", Added, S5, 5, 5)],
            vec![entry("h", Added, s6, 6, 6)],
        ]);
        assert_ne!(manifests[0].manifest_path, fixture.mx);
        assert_ne!(manifests[1].manifest_path, fixture.mf);
        assert_eq!(manifests[2], fixture.manifest(&fixture.mg).await);
        for written in [&manifests[0], &manifests[1], &manifests[3]] {
            assert_eq!(written.added_snapshot_id, s6);
        }

        assert_eq!(
            live_files(&table).await,
            ["b", "d", "g", "h"].map(String::from).into()
        );
        let summary = &snapshot.summary().additional_properties;
        assert_eq!(summary["total-records"], "106");
        assert_eq!(summary["total-data-files"], "4");
    }

    #[tokio::test]
    async fn copied_entries_keep_sequences() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;
        let rewritten = &manifest_files(&table).await[0];

        let kept: Vec<_> = entries(&table, rewritten)
            .await
            .into_iter()
            .filter(|e| e.1 != Deleted)
            .collect();

        assert_eq!(kept, vec![
            entry("b", Existing, S1, 1, 1),
            entry("d", Existing, S3, 3, 3),
        ]);
    }

    #[tokio::test]
    async fn removed_inputs_recorded() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;
        let snapshot = table.metadata().current_snapshot().unwrap();
        let s6 = snapshot.snapshot_id();

        let mut tombstones = Vec::new();
        for manifest in manifest_files(&table).await {
            tombstones.extend(
                entries(&table, &manifest)
                    .await
                    .into_iter()
                    .filter(|e| e.1 == Deleted),
            );
        }

        assert_eq!(tombstones, vec![
            entry("a", Deleted, s6, 1, 1),
            entry("f", Deleted, s6, 4, 4),
        ]);
        let summary = &snapshot.summary().additional_properties;
        assert_eq!(summary["deleted-data-files"], "2");
        assert_eq!(summary["deleted-records"], "17");
        assert_eq!(summary["added-records"], "64");
    }

    #[tokio::test]
    async fn tombstone_not_resurrected() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;

        for manifest in manifest_files(&table).await {
            let names: Vec<_> = entries(&table, &manifest)
                .await
                .into_iter()
                .map(|e| e.0)
                .collect();
            assert!(!names.contains(&"c".to_string()), "c copied: {names:?}");
        }
        assert!(!live_files(&table).await.contains("c"));
    }

    #[tokio::test]
    async fn prepare_reads_only_touched_manifests() {
        let fixture = Fixture::new().await;
        let manifest_list = fixture
            .base
            .metadata()
            .current_snapshot()
            .unwrap()
            .manifest_list()
            .to_string();

        fixture
            .replace(vec![(&fixture.mf, data("f", 16))], vec![data("h", 64)])
            .await
            .unwrap();

        let reads = fixture.log.lock().unwrap().reads.clone();
        let relative = |path: &str| {
            path.trim_start_matches("memory:")
                .trim_start_matches('/')
                .to_string()
        };
        assert_eq!(reads, vec![relative(&manifest_list), relative(&fixture.mf)]);
    }

    #[tokio::test]
    async fn delete_files_refused() {
        let fixture = Fixture::new().await;
        let deletes = file("pos", 1, DataContentType::PositionDeletes);

        let removed = fixture
            .replace(vec![(&fixture.mf, deletes.clone())], vec![data("h", 64)])
            .await;
        assert_refused(&fixture, removed, ErrorKind::FeatureUnsupported).await;

        let added = fixture
            .replace(vec![(&fixture.mf, data("f", 16))], vec![deletes.clone()])
            .await;
        assert_refused(&fixture, added, ErrorKind::FeatureUnsupported).await;

        let with_deletes = Fixture {
            base: append(fixture.base.clone(), 1006, vec![deletes]).await,
            ..fixture
        };
        let base_deletes = with_deletes
            .replace(vec![(&with_deletes.mf, data("f", 16))], vec![data("h", 64)])
            .await;
        assert_refused(&with_deletes, base_deletes, ErrorKind::FeatureUnsupported).await;
    }

    #[tokio::test]
    async fn non_live_input_refused() {
        let fixture = Fixture::new().await;
        let cases = [
            // A prior tombstone.
            vec![(fixture.mx.as_str(), data("c", 4))],
            // Live, but in another manifest.
            vec![(fixture.mf.as_str(), data("a", 1))],
            // Never in the table.
            vec![(fixture.mf.as_str(), data("z", 1))],
            // A manifest the base does not list.
            vec![("memory:///t/metadata/gone.avro", data("f", 16))],
            // Live, but a different descriptor.
            vec![(fixture.mf.as_str(), data("f", 15))],
            // One live input with one that is not.
            vec![
                (fixture.mf.as_str(), data("f", 16)),
                (fixture.mx.as_str(), data("c", 4)),
            ],
        ];

        for removed in cases {
            let result = fixture.replace(removed, vec![data("h", 64)]).await;
            assert_refused(&fixture, result, ErrorKind::PreconditionFailed).await;
        }

        let empty = Fixture {
            base: empty_table(fixture.base.file_io().clone(), FormatVersion::V2),
            ..fixture
        };
        let result = empty
            .replace(vec![(&empty.mf, data("f", 16))], vec![data("h", 64)])
            .await;
        assert_refused(&empty, result, ErrorKind::PreconditionFailed).await;
    }

    #[tokio::test]
    async fn input_live_twice_refused() {
        let fixture = Fixture::new().await;
        let tx = Transaction::new(&fixture.base);
        let tx = tx
            .fast_append()
            .set_check_duplicate(false)
            .add_data_files([data("f", 16), data("x", 1)])
            .apply(tx)
            .unwrap();
        let base = commit(fixture.base.clone(), tx).await.unwrap();
        let second = manifest_files(&base).await.pop().unwrap().manifest_path;
        let fixture = Fixture { base, ..fixture };

        let result = fixture
            .replace(
                vec![(&fixture.mf, data("f", 16)), (&second, data("x", 1))],
                vec![data("h", 64)],
            )
            .await;

        assert_refused(&fixture, result, ErrorKind::DataInvalid).await;
    }

    fn spec_file(name: &str, spec_id: i32, partition: Vec<Option<Literal>>) -> DataFile {
        DataFileBuilder::default()
            .partition_spec_id(spec_id)
            .content(DataContentType::Data)
            .file_path(format!("memory:///t/data/{name}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::from_iter(partition))
            .build()
            .unwrap()
    }

    /// Adds `spec` and makes it the default.
    fn evolve(table: Table, spec: UnboundPartitionSpec) -> Table {
        TableCommit::builder()
            .ident(table.identifier().clone())
            .updates(vec![
                TableUpdate::AddSpec { spec },
                TableUpdate::SetDefaultSpec { spec_id: -1 },
            ])
            .requirements(vec![])
            .build()
            .apply(table)
            .unwrap()
    }

    fn identity_spec() -> UnboundPartitionSpec {
        UnboundPartitionSpec::builder()
            .add_partition_field(1, "id_p", Transform::Identity)
            .unwrap()
            .build()
    }

    #[tokio::test]
    async fn touched_old_spec_keeps_spec_after_evolution() {
        let fixture = Fixture::new().await;
        let base = evolve(fixture.base.clone(), identity_spec());
        assert_eq!(base.metadata().default_partition_spec_id(), 1);
        let fixture = Fixture { base, ..fixture };
        let output = spec_file("h", 1, vec![Some(Literal::long(7))]);

        let table = fixture
            .replace(vec![(&fixture.mx, data("a", 1))], vec![output])
            .await
            .unwrap();

        let manifests = manifest_files(&table).await;
        let specs: Vec<_> = manifests.iter().map(|m| m.partition_spec_id).collect();
        assert_eq!(specs, vec![0, 0, 0, 1]);
        let kept: Vec<_> = entries(&table, &manifests[0])
            .await
            .into_iter()
            .filter(|e| e.1 != Deleted)
            .collect();
        assert_eq!(kept, vec![
            entry("b", Existing, S1, 1, 1),
            entry("d", Existing, S3, 3, 3),
        ]);
        assert_eq!(
            live_files(&table).await,
            ["b", "d", "f", "g", "h"].map(String::from).into()
        );
    }

    /// The input's spec (one field) is no longer the default (two fields).
    #[tokio::test]
    async fn removed_input_under_older_partitioned_spec_refused() {
        let fixture = Fixture::new().await;
        let base = evolve(fixture.base.clone(), identity_spec());
        let p1 = spec_file("p1", 1, vec![Some(Literal::long(7))]);
        let base = append(base, 1006, vec![p1.clone()]).await;
        let p1_manifest = manifest_files(&base).await.pop().unwrap().manifest_path;
        let spec2 = UnboundPartitionSpec::builder()
            .add_partition_field(1, "id_p", Transform::Identity)
            .unwrap()
            .add_partition_field(1, "id_b", Transform::Bucket(4))
            .unwrap()
            .build();
        let base = evolve(base, spec2);
        let fixture = Fixture { base, ..fixture };
        let output = spec_file("h", 2, vec![Some(Literal::long(7)), Some(Literal::int(1))]);

        let result = fixture
            .replace(vec![(&p1_manifest, p1)], vec![output])
            .await;

        assert_refused(&fixture, result, ErrorKind::FeatureUnsupported).await;
        assert_eq!(fixture.log.lock().unwrap().reads, Vec::<String>::new());
    }

    #[tokio::test]
    async fn incomplete_input_refused() {
        let fixture = Fixture::new().await;
        let cases = [
            (vec![], vec![data("h", 64)]),
            (
                vec![
                    (fixture.mf.as_str(), data("f", 16)),
                    (fixture.mf.as_str(), data("f", 16)),
                ],
                vec![data("h", 64)],
            ),
            (vec![(fixture.mf.as_str(), data("f", 16))], vec![
                data("h", 64),
                data("h", 64),
            ]),
            (vec![(fixture.mf.as_str(), data("f", 16))], vec![data(
                "f", 16,
            )]),
            // An output path already live in a touched manifest.
            (vec![(fixture.mx.as_str(), data("a", 1))], vec![data(
                "b", 2,
            )]),
        ];

        for (removed, added) in cases {
            let result = fixture.replace(removed, added).await;
            assert_refused(&fixture, result, ErrorKind::DataInvalid).await;
        }
    }

    #[tokio::test]
    async fn non_v2_table_refused() {
        for format_version in [FormatVersion::V1, FormatVersion::V3] {
            let fixture = Fixture::new().await;
            let other = Fixture {
                base: empty_table(fixture.base.file_io().clone(), format_version),
                ..fixture
            };
            let result = other
                .replace(vec![(&other.mf, data("f", 16))], vec![data("h", 64)])
                .await;
            assert_refused(&other, result, ErrorKind::FeatureUnsupported).await;
        }
    }

    #[tokio::test]
    async fn prepare_replace_commits_through_exact_base() {
        let catalog = crate::catalog::memory::tests::new_memory_catalog().await;
        let namespace = crate::NamespaceIdent::new("db".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let table = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(
                        Schema::builder()
                            .with_fields([NestedField::required(
                                1,
                                "id",
                                Type::Primitive(PrimitiveType::Long),
                            )
                            .into()])
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .await
            .unwrap();
        let tx = Transaction::new(&table);
        let table = tx
            .fast_append()
            .add_data_files([data("a", 1), data("b", 2)])
            .apply(tx)
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        let manifest = manifest_files(&table).await.remove(0).manifest_path;

        let base = catalog.load_table_exact(table.identifier()).await.unwrap();
        let outcome = Transaction::new(base.table())
            .prepare_replace([(manifest, data("a", 1))], [data("h", 4)])
            .commit_exact_base(&catalog, base)
            .await
            .unwrap();

        let crate::ExactCommitOutcome::Committed(committed) = outcome else {
            panic!("replace must commit");
        };
        assert_eq!(
            live_files(&committed).await,
            ["b", "h"].map(String::from).into()
        );
    }
}
