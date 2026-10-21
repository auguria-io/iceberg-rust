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

//! A replace commit prepared from the manifests that hold its inputs.
//!
//! [`prepare_replace`] reads the base manifest list and only the manifests
//! that hold a removed input. It writes the new manifests and manifest list
//! and returns a [`PreparedReplace`]; it sends nothing.
//! [`PreparedReplace::stage`] hands the commit to
//! [`Catalog::stage_table_exact`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uuid::Uuid;

use super::Transaction;
use super::snapshot::{DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestEntry,
    ManifestFile, Operation,
};
use crate::table::Table;
use crate::{
    Catalog, Error, ErrorKind, ExactCommitError, ExactCommitResult, ExactTableBase,
    PreparedExactCommit, Result, TableCommit,
};

/// The manifests a caller's view lists for the base snapshot.
///
/// Build it from the caller's own index, never from the base: the replace
/// refuses unless it equals the base manifest list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestView {
    /// The snapshot the view describes; must be the base's main snapshot.
    pub snapshot_id: i64,
    /// Every manifest the view lists for that snapshot, in any order.
    pub manifests: Vec<ViewManifest>,
}

/// One manifest of a [`ManifestView`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewManifest {
    /// Manifest path, exactly as the manifest list records it.
    pub path: String,
    /// Manifest length in bytes, as the manifest list records it.
    pub length: i64,
    /// Data or delete manifest.
    pub content: ManifestContentType,
    /// Partition spec of the manifest.
    pub partition_spec_id: i32,
}

/// A data file to remove, with the base manifest that lists it live and the
/// sequence numbers the caller saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaceInput {
    /// Path of the base manifest that lists the file live.
    pub manifest_path: String,
    /// The file's full descriptor, equal to its live entry.
    pub data_file: DataFile,
    /// The live entry's data sequence number.
    pub data_sequence_number: i64,
    /// The live entry's file sequence number.
    pub file_sequence_number: i64,
}

/// A replace commit whose manifests and manifest list are written and which
/// nothing has staged or sent.
#[derive(Debug)]
pub struct PreparedReplace {
    base: Table,
    commit: TableCommit,
    snapshot_id: i64,
    sequence_number: i64,
}

impl PreparedReplace {
    /// The new snapshot's id.
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// The new snapshot's sequence number.
    pub fn sequence_number(&self) -> i64 {
        self.sequence_number
    }

    /// Stages this commit against `base`, which must be the exact base it was
    /// prepared from. Sends nothing; see [`Catalog::stage_table_exact`].
    pub async fn stage(
        self,
        catalog: &dyn Catalog,
        base: ExactTableBase,
    ) -> ExactCommitResult<PreparedExactCommit> {
        if !Transaction::matches_exact_base(&self.base, base.table()) {
            return Err(ExactCommitError::before_cas(stale(
                "Prepared replace was built from another exact base",
            )));
        }
        catalog.stage_table_exact(base, self.commit).await
    }
}

/// Prepares a commit that replaces exactly `inputs` with `outputs` on the
/// main branch of `base`.
///
/// Reads the base manifest list, then each manifest named by an input, once.
/// It refuses, before writing anything, unless:
/// - `view` lists exactly the base snapshot's manifests;
/// - every input is live, once, in the manifest it names, with an equal
///   descriptor and the given sequence numbers;
/// - no output path is live in a manifest it reads;
/// - the table is format version 2, lists no delete manifest, and every input
///   and output is a data file under the default partition spec.
///
/// The new snapshot carries untouched manifests by reference. Each touched
/// manifest is rewritten: its other live entries as `Existing` with their
/// snapshot ids and sequence numbers, its inputs as `Deleted` under the new
/// snapshot, and none of its older `Deleted` entries. The outputs go in one
/// new manifest as `Added`, inheriting the new snapshot id and sequence
/// number.
///
/// Files in manifests it does not read are not checked: an input also live
/// in an untouched manifest, or an output path live there, is the view's
/// responsibility.
pub async fn prepare_replace(
    base: &ExactTableBase,
    view: &ManifestView,
    inputs: Vec<ReplaceInput>,
    outputs: Vec<DataFile>,
) -> Result<PreparedReplace> {
    let table = base.table();
    check_request(table, &inputs, &outputs)?;
    let snapshot_id = SnapshotProducer::generate_unique_snapshot_id(table);
    let sequence_number = table.metadata().next_sequence_number();
    let mut producer = SnapshotProducer::new(
        table,
        Uuid::now_v7(),
        None,
        Some(snapshot_id),
        HashMap::new(),
        outputs.clone(),
        Vec::new(),
        inputs.iter().map(|i| i.data_file.clone()).collect(),
        Vec::new(),
    );
    producer.validate_added_data_files(&outputs)?;
    let plan = plan(table, view, &inputs, &outputs).await?;

    let mut manifests = Vec::with_capacity(plan.len());
    for planned in plan {
        manifests.push(match planned {
            Planned::Carried(manifest) => manifest,
            Planned::Rewritten {
                spec_id,
                kept,
                removed,
            } => {
                let mut writer =
                    producer.new_manifest_writer(ManifestContentType::Data, spec_id)?;
                for entry in kept {
                    writer.add_existing_entry(entry)?;
                }
                for entry in removed {
                    writer.add_delete_entry(entry)?;
                }
                writer.write_manifest_file().await?
            }
        });
    }
    let mut action = producer
        .commit(PlannedManifests(manifests), DefaultManifestProcess)
        .await?;

    Ok(PreparedReplace {
        base: table.clone(),
        commit: TableCommit::builder()
            .ident(table.identifier().clone())
            .updates(action.take_updates())
            .requirements(action.take_requirements())
            .build(),
        snapshot_id,
        sequence_number,
    })
}

/// Checks the request against the base metadata, without reading files.
fn check_request(table: &Table, inputs: &[ReplaceInput], outputs: &[DataFile]) -> Result<()> {
    if inputs.is_empty() || outputs.is_empty() {
        return Err(invalid("Replace needs at least one input and one output"));
    }
    let metadata = table.metadata();
    if metadata.format_version() != FormatVersion::V2 {
        return Err(unsupported(format!(
            "Replace supports format version 2 tables only, not {:?}",
            metadata.format_version()
        )));
    }
    let default_spec = metadata.default_partition_spec_id();
    let mut paths = HashSet::new();
    for file in inputs.iter().map(|i| &i.data_file).chain(outputs) {
        let path = file.file_path();
        if file.content_type() != DataContentType::Data {
            return Err(unsupported(format!(
                "Replace supports data files only: {path} is {:?}",
                file.content_type()
            )));
        }
        if file.partition_spec_id != default_spec {
            return Err(unsupported(format!(
                "Replace file {path} is under spec {}, not the default spec {default_spec}",
                file.partition_spec_id
            )));
        }
        if !paths.insert(path) {
            return Err(invalid(format!("Replace lists file {path} more than once")));
        }
    }
    Ok(())
}

/// The new snapshot's manifests before the output manifest, in base order.
enum Planned {
    /// An untouched base manifest, carried by reference.
    Carried(ManifestFile),
    /// A touched base manifest: its other live entries and its inputs.
    Rewritten {
        spec_id: i32,
        kept: Vec<ManifestEntry>,
        removed: Vec<ManifestEntry>,
    },
}

/// Reads the base manifest list and the touched manifests, and checks the
/// view and every input against them.
async fn plan(
    table: &Table,
    view: &ManifestView,
    inputs: &[ReplaceInput],
    outputs: &[DataFile],
) -> Result<Vec<Planned>> {
    let metadata = table.metadata();
    let snapshot = metadata
        .snapshot_for_ref(MAIN_BRANCH)
        .ok_or_else(|| stale("Replace base has no main snapshot"))?;
    if snapshot.snapshot_id() != view.snapshot_id {
        return Err(stale(format!(
            "View describes snapshot {}, the base is snapshot {}",
            view.snapshot_id,
            snapshot.snapshot_id()
        )));
    }
    let list = snapshot
        .load_manifest_list(table.file_io(), metadata)
        .await?;
    check_view(list.entries(), view)?;
    if let Some(manifest) = list
        .entries()
        .iter()
        .find(|m| m.content == ManifestContentType::Deletes)
    {
        return Err(unsupported(format!(
            "Replace does not support tables with delete manifests: {}",
            manifest.manifest_path
        )));
    }

    let mut wanted: HashMap<&str, HashMap<&str, &ReplaceInput>> = HashMap::new();
    for input in inputs {
        wanted
            .entry(input.manifest_path.as_str())
            .or_default()
            .insert(input.data_file.file_path(), input);
    }
    let input_paths: HashSet<&str> = inputs.iter().map(|i| i.data_file.file_path()).collect();
    let output_paths: HashSet<&str> = outputs.iter().map(|f| f.file_path()).collect();

    let mut planned = Vec::with_capacity(list.entries().len());
    for manifest in list.consume_entries() {
        let Some(mut here) = wanted.remove(manifest.manifest_path.as_str()) else {
            planned.push(Planned::Carried(manifest));
            continue;
        };
        let (entries, _) = manifest.load_manifest(table.file_io()).await?.into_parts();
        let mut kept = Vec::new();
        let mut removed = Vec::new();
        for entry in entries {
            let path = entry.file_path();
            if entry.content_type() != DataContentType::Data {
                return Err(invalid(format!(
                    "Data manifest {} lists {path} as {:?}",
                    manifest.manifest_path,
                    entry.content_type()
                )));
            }
            // Older tombstones are dropped, never copied, so none can return.
            if !entry.is_alive() {
                continue;
            }
            if output_paths.contains(path) {
                return Err(invalid(format!("Replace output {path} is already live")));
            }
            match here.remove(path) {
                Some(input) if input.matches(&entry) => removed.push(Arc::unwrap_or_clone(entry)),
                Some(_) => {
                    return Err(stale(format!(
                        "Replace input {path} differs from its live entry in {}",
                        manifest.manifest_path
                    )));
                }
                None if input_paths.contains(path) => {
                    return Err(invalid(format!(
                        "Replace input {path} is live more than once"
                    )));
                }
                None if entry.sequence_number().is_none()
                    || entry.file_sequence_number().is_none() =>
                {
                    return Err(invalid(format!(
                        "Live entry {path} in {} has no resolved sequence numbers",
                        manifest.manifest_path
                    )));
                }
                None => kept.push(Arc::unwrap_or_clone(entry)),
            }
        }
        if let Some(path) = here.into_keys().next() {
            return Err(stale(format!(
                "Replace input {path} is not live in manifest {}",
                manifest.manifest_path
            )));
        }
        planned.push(Planned::Rewritten {
            spec_id: manifest.partition_spec_id,
            kept,
            removed,
        });
    }
    if let Some(manifest) = wanted.into_keys().next() {
        return Err(stale(format!(
            "Replace input manifest {manifest} is not in the base snapshot"
        )));
    }
    Ok(planned)
}

/// Refuses unless `view` lists exactly the manifests of `list`.
fn check_view(list: &[ManifestFile], view: &ManifestView) -> Result<()> {
    let key = |path: &str, length: i64, content: ManifestContentType, spec: i32| {
        (path.to_string(), length, content as i32, spec)
    };
    let mut base: Vec<_> = list
        .iter()
        .map(|m| {
            key(
                &m.manifest_path,
                m.manifest_length,
                m.content,
                m.partition_spec_id,
            )
        })
        .collect();
    let mut seen: Vec<_> = view
        .manifests
        .iter()
        .map(|m| key(&m.path, m.length, m.content, m.partition_spec_id))
        .collect();
    base.sort_unstable();
    seen.sort_unstable();
    if base != seen {
        return Err(stale(format!(
            "View of snapshot {} does not list the base manifests",
            view.snapshot_id
        )));
    }
    Ok(())
}

impl ReplaceInput {
    fn matches(&self, entry: &ManifestEntry) -> bool {
        entry.data_file() == &self.data_file
            && entry.sequence_number() == Some(self.data_sequence_number)
            && entry.file_sequence_number() == Some(self.file_sequence_number)
    }
}

/// Supplies the planned manifests; the inputs are already recorded in them.
struct PlannedManifests(Vec<ManifestFile>);

impl SnapshotProduceOperation for PlannedManifests {
    fn operation(&self) -> Operation {
        Operation::Replace
    }

    async fn delete_entries(&self, _: &SnapshotProducer<'_>) -> Result<Vec<ManifestEntry>> {
        Ok(Vec::new())
    }

    async fn existing_manifest(&self, _: &mut SnapshotProducer<'_>) -> Result<Vec<ManifestFile>> {
        Ok(self.0.clone())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataInvalid, message)
}

fn stale(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::PreconditionFailed, message)
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::FeatureUnsupported, message)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use futures::TryStreamExt;

    use super::*;
    use crate::io::FileIOBuilder;
    use crate::spec::ManifestStatus::{self, Added, Deleted, Existing};
    use crate::spec::{
        DataFileBuilder, DataFileFormat, Literal, NestedField, PrimitiveType, Schema, Struct,
        TableMetadataBuilder, Transform, Type, UnboundPartitionSpec,
    };
    use crate::transaction::ApplyTransactionAction;
    use crate::{MetadataLocation, TableCreation, TableIdent, TableUpdate};

    const S1: i64 = 1001;
    const S2: i64 = 1002;
    const S3: i64 = 1003;
    const S4: i64 = 1004;
    const S5: i64 = 1005;
    const S6: i64 = 1006;

    /// Every fixture file is 100 bytes, in partition day 7 of spec 0.
    fn data(name: &str, rows: u64) -> DataFile {
        file(name, rows, DataContentType::Data)
    }

    fn file(name: &str, rows: u64, content: DataContentType) -> DataFile {
        let mut builder = DataFileBuilder::default();
        builder
            .partition_spec_id(0)
            .content(content)
            .file_path(format!("memory:///t/data/{name}.parquet"))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(rows)
            .partition(Struct::from_iter([Some(Literal::int(7))]));
        if content == DataContentType::EqualityDeletes {
            builder.equality_ids(Some(vec![1]));
        }
        builder.build().unwrap()
    }

    fn name(path: &str) -> String {
        path.trim_start_matches("memory:///t/data/")
            .trim_end_matches(".parquet")
            .to_string()
    }

    /// A manifest entry as loaded: file, status, snapshot id, data sequence,
    /// file sequence, rows.
    type Entry = (String, ManifestStatus, i64, i64, i64, u64);

    fn entry(
        n: &str,
        status: ManifestStatus,
        snap: i64,
        seq: i64,
        file_seq: i64,
        rows: u64,
    ) -> Entry {
        (n.to_string(), status, snap, seq, file_seq, rows)
    }

    fn empty_table(format_version: FormatVersion) -> Table {
        let schema = Schema::builder()
            .with_fields([
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::required(2, "day", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap();
        let spec = UnboundPartitionSpec::builder()
            .add_partition_field(2, "day", Transform::Identity)
            .unwrap()
            .build();
        let metadata = TableMetadataBuilder::from_table_creation(
            TableCreation::builder()
                .name("t".to_string())
                .location("memory:///t".to_string())
                .schema(schema)
                .partition_spec(spec)
                .format_version(format_version)
                .build(),
        )
        .unwrap()
        .build()
        .unwrap()
        .metadata;
        Table::builder()
            .file_io(FileIOBuilder::new("memory").build().unwrap())
            .metadata(metadata)
            .metadata_location(MetadataLocation::new_with_table_location("memory:///t").to_string())
            .identifier(TableIdent::from_strs(["db", "t"]).unwrap())
            .build()
            .unwrap()
    }

    /// Applies a transaction as a catalog would, without one.
    async fn commit(table: Table, tx: Transaction) -> Table {
        let (updates, requirements) = tx.apply_actions().await.unwrap();
        TableCommit::builder()
            .ident(table.identifier().clone())
            .updates(updates)
            .requirements(requirements)
            .build()
            .apply(table)
            .unwrap()
    }

    async fn append(table: Table, snapshot_id: i64, files: Vec<DataFile>) -> Table {
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .set_snapshot_id(snapshot_id)
            .add_data_files(files)
            .apply(tx)
            .unwrap();
        commit(table, tx).await
    }

    async fn list(table: &Table) -> Vec<ManifestFile> {
        let snapshot = table.metadata().current_snapshot().unwrap();
        snapshot
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
                    e.record_count(),
                )
            })
            .collect();
        entries.sort_by_key(|e| (e.0.clone(), e.2, e.1 as i32));
        entries
    }

    /// File name to data sequence number, from `plan_files`, an independent
    /// reader of the committed snapshot.
    async fn planned(table: &Table) -> BTreeMap<String, i64> {
        table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .map_ok(|task| (name(task.data_file_path()), task.sequence_number))
            .try_collect()
            .await
            .unwrap()
    }

    fn view_of(snapshot_id: i64, manifests: &[ManifestFile]) -> ManifestView {
        ManifestView {
            snapshot_id,
            manifests: manifests
                .iter()
                .map(|m| ViewManifest {
                    path: m.manifest_path.clone(),
                    length: m.manifest_length,
                    content: m.content,
                    partition_spec_id: m.partition_spec_id,
                })
                .collect(),
        }
    }

    fn input(manifest: &ManifestFile, file: DataFile, seq: i64, file_seq: i64) -> ReplaceInput {
        ReplaceInput {
            manifest_path: manifest.manifest_path.clone(),
            data_file: file,
            data_sequence_number: seq,
            file_sequence_number: file_seq,
        }
    }

    /// The story, one snapshot per step:
    /// - S1 appends a (1 row) and b (2): manifest M1.
    /// - S2 appends c (4): manifest M2.
    /// - S3 merges M1 and M2 while rewriting c into d (8): one manifest Mx
    ///   with a and b `Existing` (S1, seq 1), c `Deleted` (S3, seq 2), d
    ///   `Added` (S3, seq 3).
    /// - S4 appends f (16): Mf. S5 appends g (32): Mg.
    ///
    /// Base S5 is sequence 5 and holds a, b, d, f and g: 59 rows, 5 files.
    struct Fixture {
        base: ExactTableBase,
        mx: ManifestFile,
        mf: ManifestFile,
        mg: ManifestFile,
    }

    impl Fixture {
        async fn new() -> Self {
            let table = empty_table(FormatVersion::V2);
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
            let table = commit(table, rewrite.apply(tx).unwrap()).await;
            let table = append(table, S4, vec![data("f", 16)]).await;
            let table = append(table, S5, vec![data("g", 32)]).await;

            let [mx, mf, mg] = list(&table).await.try_into().unwrap();
            assert_eq!(
                entries(&table, &mx).await,
                vec![
                    entry("a", Existing, S1, 1, 1, 1),
                    entry("b", Existing, S1, 1, 1, 2),
                    entry("c", Deleted, S3, 2, 2, 4),
                    entry("d", Added, S3, 3, 3, 8),
                ],
                "fixture story: Mx"
            );
            assert_eq!(entries(&table, &mf).await, vec![entry(
                "f", Added, S4, 4, 4, 16
            )]);
            assert_eq!(entries(&table, &mg).await, vec![entry(
                "g", Added, S5, 5, 5, 32
            )]);
            Self {
                base: ExactTableBase::new(table, ()),
                mx,
                mf,
                mg,
            }
        }

        fn table(&self) -> &Table {
            self.base.table()
        }

        fn view(&self) -> ManifestView {
            view_of(S5, &[self.mx.clone(), self.mf.clone(), self.mg.clone()])
        }

        /// Prepares on the base with its true view, then applies the commit
        /// as a catalog would.
        async fn replace(
            &self,
            inputs: Vec<ReplaceInput>,
            outputs: Vec<DataFile>,
        ) -> Result<Table> {
            let prepared = prepare_replace(&self.base, &self.view(), inputs, outputs).await?;
            prepared.commit.apply(self.table().clone())
        }

        /// Removes a (in Mx) and f (in Mf); adds h (64 rows).
        async fn standard(&self) -> Table {
            self.replace(
                vec![
                    input(&self.mx, data("a", 1), 1, 1),
                    input(&self.mf, data("f", 16), 4, 4),
                ],
                vec![data("h", 64)],
            )
            .await
            .unwrap()
        }

        /// Paths of every object under the table, for "wrote nothing" checks.
        async fn objects(&self) -> BTreeSet<String> {
            self.table()
                .file_io()
                .list("memory:///t/", true)
                .await
                .unwrap()
                .map_ok(|e| e.path)
                .try_collect()
                .await
                .unwrap()
        }
    }

    async fn assert_refused(
        fixture: &Fixture,
        view: &ManifestView,
        inputs: Vec<ReplaceInput>,
        outputs: Vec<DataFile>,
        kind: ErrorKind,
    ) {
        let before = fixture.objects().await;
        let error = prepare_replace(&fixture.base, view, inputs, outputs)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), kind, "{error}");
        assert_eq!(fixture.objects().await, before, "a refusal wrote files");
    }

    /// The expected result is written out from the fixture story above, not
    /// computed by any replace or rewrite code, and the live set is read back
    /// with `plan_files`.
    #[tokio::test]
    async fn prepare_equivalence() {
        let fixture = Fixture::new().await;
        let prepared = prepare_replace(
            &fixture.base,
            &fixture.view(),
            vec![
                input(&fixture.mx, data("a", 1), 1, 1),
                input(&fixture.mf, data("f", 16), 4, 4),
            ],
            vec![data("h", 64)],
        )
        .await
        .unwrap();
        let s6 = prepared.snapshot_id();
        assert_eq!(prepared.sequence_number(), 6);
        let mut commit = prepared.commit;
        let requirements = commit.take_requirements();
        let updates = commit.take_updates();
        let table = TableCommit::builder()
            .ident(fixture.table().identifier().clone())
            .updates(updates.clone())
            .requirements(requirements.clone())
            .build()
            .apply(fixture.table().clone())
            .unwrap();

        // Snapshot fields.
        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.snapshot_id(), s6);
        assert!(s6 > 0 && ![S1, S2, S3, S4, S5].contains(&s6));
        assert_eq!(snapshot.parent_snapshot_id(), Some(S5));
        assert_eq!(snapshot.sequence_number(), 6);
        assert_eq!(table.metadata().last_sequence_number(), 6);
        assert_eq!(table.metadata().uuid(), fixture.table().metadata().uuid());

        // Manifest list: Mx and Mf rewritten in place, Mg carried, then the
        // output manifest.
        let manifests = list(&table).await;
        assert_eq!(manifests.len(), 4);
        assert_eq!(manifests[2], fixture.mg, "Mg is carried unchanged");
        for (i, written) in [0, 1, 3].map(|i| (i, &manifests[i])) {
            assert_eq!(written.added_snapshot_id, s6, "manifest {i}");
            assert_eq!(written.sequence_number, 6, "manifest {i}");
            assert_eq!(written.content, ManifestContentType::Data);
            assert_eq!(written.partition_spec_id, 0);
        }
        assert!(
            ![&fixture.mx, &fixture.mf, &fixture.mg]
                .iter()
                .any(|m| m.manifest_path == manifests[0].manifest_path
                    || m.manifest_path == manifests[1].manifest_path
                    || m.manifest_path == manifests[3].manifest_path)
        );
        // Counts and the least live data sequence of each written manifest.
        let counts = |m: &ManifestFile| {
            (
                m.added_files_count,
                m.existing_files_count,
                m.deleted_files_count,
                m.added_rows_count,
                m.existing_rows_count,
                m.deleted_rows_count,
                m.min_sequence_number,
            )
        };
        assert_eq!(
            counts(&manifests[0]),
            (Some(0), Some(2), Some(1), Some(0), Some(10), Some(1), 1)
        );
        assert_eq!(
            counts(&manifests[1]),
            (Some(0), Some(0), Some(1), Some(0), Some(0), Some(16), 6)
        );
        assert_eq!(
            counts(&manifests[3]),
            (Some(1), Some(0), Some(0), Some(64), Some(0), Some(0), 6)
        );

        // Every entry of every manifest.
        let mut got = Vec::new();
        for manifest in &manifests {
            got.push(entries(&table, manifest).await);
        }
        assert_eq!(got, vec![
            vec![
                entry("a", Deleted, s6, 1, 1, 1),
                entry("b", Existing, S1, 1, 1, 2),
                entry("d", Existing, S3, 3, 3, 8),
            ],
            vec![entry("f", Deleted, s6, 4, 4, 16)],
            vec![entry("g", Added, S5, 5, 5, 32)],
            vec![entry("h", Added, s6, 6, 6, 64)],
        ]);

        // Live set and data sequences, read by the scan planner.
        assert_eq!(
            planned(&table).await,
            BTreeMap::from([
                ("b".to_string(), 1),
                ("d".to_string(), 3),
                ("g".to_string(), 5),
                ("h".to_string(), 6),
            ])
        );

        // Summary: 59 - 17 + 64 = 106 rows; 5 - 2 + 1 = 4 files; sizes in
        // 100-byte files.
        let summary = &snapshot.summary();
        assert_eq!(summary.operation, Operation::Replace);
        let props: BTreeMap<_, _> = summary
            .additional_properties
            .iter()
            .filter(|(k, _)| !k.starts_with("partitions.") && *k != "changed-partition-count")
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            props,
            BTreeMap::from([
                ("added-data-files", "1"),
                ("added-records", "64"),
                ("added-files-size", "100"),
                ("deleted-data-files", "2"),
                ("deleted-records", "17"),
                ("removed-files-size", "200"),
                ("total-data-files", "4"),
                ("total-records", "106"),
                ("total-files-size", "400"),
                ("total-delete-files", "0"),
                ("total-position-deletes", "0"),
                ("total-equality-deletes", "0"),
            ])
        );

        // Updates and requirements: one snapshot, main moves from S5 to it.
        assert_eq!(requirements, vec![
            crate::TableRequirement::UuidMatch {
                uuid: fixture.table().metadata().uuid()
            },
            crate::TableRequirement::RefSnapshotIdMatch {
                r#ref: MAIN_BRANCH.to_string(),
                snapshot_id: Some(S5)
            },
        ]);
        assert_eq!(updates.len(), 2);
        assert!(
            matches!(&updates[0], TableUpdate::AddSnapshot { snapshot } if snapshot.snapshot_id() == s6)
        );
        assert!(
            matches!(&updates[1], TableUpdate::SetSnapshotRef { ref_name, reference }
            if ref_name == MAIN_BRANCH && reference.snapshot_id == s6)
        );
    }

    #[tokio::test]
    async fn copied_entries_keep_sequences() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;
        let rewritten = &list(&table).await[0];

        let kept: Vec<_> = entries(&table, rewritten)
            .await
            .into_iter()
            .filter(|e| e.1 != Deleted)
            .collect();

        // b and d keep the snapshot ids and sequences they had in Mx.
        assert_eq!(kept, vec![
            entry("b", Existing, S1, 1, 1, 2),
            entry("d", Existing, S3, 3, 3, 8),
        ]);
    }

    #[tokio::test]
    async fn removed_inputs_recorded() {
        let fixture = Fixture::new().await;
        let table = fixture.standard().await;
        let s6 = table.metadata().current_snapshot().unwrap().snapshot_id();

        let mut tombstones = Vec::new();
        for manifest in list(&table).await {
            tombstones.extend(
                entries(&table, &manifest)
                    .await
                    .into_iter()
                    .filter(|e| e.1 == Deleted),
            );
        }

        // Each input once, under the new snapshot, with its own sequences.
        assert_eq!(tombstones, vec![
            entry("a", Deleted, s6, 1, 1, 1),
            entry("f", Deleted, s6, 4, 4, 16),
        ]);
    }

    #[tokio::test]
    async fn tombstone_not_resurrected() {
        let fixture = Fixture::new().await;
        // Mx holds c as Deleted; touch Mx.
        let table = fixture.standard().await;

        for manifest in list(&table).await {
            let names: Vec<_> = entries(&table, &manifest)
                .await
                .into_iter()
                .map(|e| e.0)
                .collect();
            assert!(!names.contains(&"c".to_string()), "c copied: {names:?}");
        }
        assert!(!planned(&table).await.contains_key("c"));

        // The tombstone itself cannot be named as an input.
        assert_refused(
            &fixture,
            &fixture.view(),
            vec![input(&fixture.mx, data("c", 4), 2, 2)],
            vec![data("h", 64)],
            ErrorKind::PreconditionFailed,
        )
        .await;
    }

    #[tokio::test]
    async fn prepare_reads_only_touched_manifests() {
        let fixture = Fixture::new().await;
        let file_io = fixture.table().file_io();
        let file_name = |path: &str| path.rsplit('/').next().unwrap().to_string();
        let keep: BTreeSet<String> = [
            fixture
                .table()
                .metadata()
                .current_snapshot()
                .unwrap()
                .manifest_list(),
            fixture.mf.manifest_path.as_str(),
        ]
        .map(file_name)
        .into();
        // Delete every other object under metadata/: untouched manifests
        // (Mx, Mg), older manifests and every other manifest list. Any read
        // of them fails the prepare.
        let mut deleted = Vec::new();
        for path in fixture.objects().await {
            if path.contains("/metadata/") && !keep.contains(&file_name(&path)) {
                file_io.delete(&path).await.unwrap();
                deleted.push(file_name(&path));
            }
        }
        // The four older manifest lists are gone, with every manifest but Mf.
        let lists = deleted.iter().filter(|n| n.starts_with("snap-")).count();
        assert_eq!(lists, 4, "{deleted:?}");
        assert!(!file_io.exists(&fixture.mx.manifest_path).await.unwrap());
        assert!(!file_io.exists(&fixture.mg.manifest_path).await.unwrap());

        let prepared = prepare_replace(
            &fixture.base,
            &fixture.view(),
            vec![input(&fixture.mf, data("f", 16), 4, 4)],
            vec![data("h", 64)],
        )
        .await
        .unwrap();

        // Mx and Mg are carried as they were, by reference.
        let s6 = prepared.snapshot_id();
        let table = prepared.commit.apply(fixture.table().clone()).unwrap();
        let manifests = list(&table).await;
        assert_eq!(manifests[0], fixture.mx);
        assert_eq!(manifests[2], fixture.mg);
        assert_eq!(entries(&table, &manifests[1]).await, vec![entry(
            "f", Deleted, s6, 4, 4, 16
        )]);
    }

    #[tokio::test]
    async fn view_must_match_base_list() {
        let fixture = Fixture::new().await;
        let inputs = || vec![input(&fixture.mf, data("f", 16), 4, 4)];
        let outputs = || vec![data("h", 64)];
        let mut cases = Vec::new();
        // Another snapshot.
        cases.push(ManifestView {
            snapshot_id: S4,
            ..fixture.view()
        });
        // One manifest missing.
        let mut view = fixture.view();
        view.manifests.pop();
        cases.push(view);
        // One manifest extra.
        let mut view = fixture.view();
        view.manifests.push(ViewManifest {
            path: "memory:///t/metadata/other.avro".to_string(),
            ..view.manifests[0].clone()
        });
        cases.push(view);
        // Same path, other length.
        let mut view = fixture.view();
        view.manifests[1].length += 1;
        cases.push(view);
        // Same path, other spec.
        let mut view = fixture.view();
        view.manifests[1].partition_spec_id = 1;
        cases.push(view);
        // Same path, other content.
        let mut view = fixture.view();
        view.manifests[1].content = ManifestContentType::Deletes;
        cases.push(view);

        for view in cases {
            assert_refused(
                &fixture,
                &view,
                inputs(),
                outputs(),
                ErrorKind::PreconditionFailed,
            )
            .await;
        }
        // Order does not matter.
        let mut view = fixture.view();
        view.manifests.reverse();
        prepare_replace(&fixture.base, &view, inputs(), outputs())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn stale_input_refused() {
        let fixture = Fixture::new().await;
        let cases = [
            // Live, but in another manifest.
            vec![input(&fixture.mf, data("a", 1), 1, 1)],
            // Never in the table.
            vec![input(&fixture.mf, data("z", 1), 4, 4)],
            // A manifest the base does not list.
            vec![ReplaceInput {
                manifest_path: "memory:///t/metadata/gone.avro".to_string(),
                ..input(&fixture.mf, data("f", 16), 4, 4)
            }],
            // Live, other descriptor.
            vec![input(&fixture.mf, data("f", 15), 4, 4)],
            // Live, other data sequence.
            vec![input(&fixture.mf, data("f", 16), 3, 4)],
            // Live, other file sequence.
            vec![input(&fixture.mf, data("f", 16), 4, 3)],
            // One good input and one tombstone.
            vec![
                input(&fixture.mf, data("f", 16), 4, 4),
                input(&fixture.mx, data("c", 4), 2, 2),
            ],
        ];
        for inputs in cases {
            assert_refused(
                &fixture,
                &fixture.view(),
                inputs,
                vec![data("h", 64)],
                ErrorKind::PreconditionFailed,
            )
            .await;
        }

        let empty = Fixture {
            base: ExactTableBase::new(empty_table(FormatVersion::V2), ()),
            ..fixture
        };
        assert_refused(
            &empty,
            &empty.view(),
            vec![input(&empty.mf, data("f", 16), 4, 4)],
            vec![data("h", 64)],
            ErrorKind::PreconditionFailed,
        )
        .await;
    }

    #[tokio::test]
    async fn bad_request_refused() {
        let fixture = Fixture::new().await;
        let f = || input(&fixture.mf, data("f", 16), 4, 4);
        let cases = [
            (vec![], vec![data("h", 64)]),
            (vec![f()], vec![]),
            (vec![f(), f()], vec![data("h", 64)]),
            (vec![f()], vec![data("h", 64), data("h", 64)]),
            (vec![f()], vec![data("f", 16)]),
            // An output already live in a touched manifest.
            (vec![input(&fixture.mx, data("a", 1), 1, 1)], vec![data(
                "b", 2,
            )]),
        ];
        for (inputs, outputs) in cases {
            assert_refused(
                &fixture,
                &fixture.view(),
                inputs,
                outputs,
                ErrorKind::DataInvalid,
            )
            .await;
        }

        // An input live twice in its manifest.
        let tx = Transaction::new(fixture.table());
        let tx = tx
            .fast_append()
            .set_snapshot_id(S6)
            .set_check_duplicate(false)
            .add_data_files([data("x", 1), data("x", 1)])
            .apply(tx)
            .unwrap();
        let table = commit(fixture.table().clone(), tx).await;
        let manifests = list(&table).await;
        let twice = Fixture {
            base: ExactTableBase::new(table, ()),
            ..fixture
        };
        assert_refused(
            &twice,
            &view_of(S6, &manifests),
            vec![input(&manifests[3], data("x", 1), 6, 6)],
            vec![data("h", 64)],
            ErrorKind::DataInvalid,
        )
        .await;
    }

    #[tokio::test]
    async fn unsupported_shapes_refused() {
        let fixture = Fixture::new().await;
        let f = || input(&fixture.mf, data("f", 16), 4, 4);

        // Delete files as input or output.
        let deletes = file("eq", 1, DataContentType::EqualityDeletes);
        let positional = file("pos", 1, DataContentType::PositionDeletes);
        let cases = [
            (vec![input(&fixture.mf, deletes.clone(), 4, 4)], vec![data(
                "h", 64,
            )]),
            (vec![f()], vec![deletes.clone()]),
            (vec![f()], vec![positional]),
        ];
        for (inputs, outputs) in cases {
            assert_refused(
                &fixture,
                &fixture.view(),
                inputs,
                outputs,
                ErrorKind::FeatureUnsupported,
            )
            .await;
        }

        // A base that lists a delete manifest.
        let with_deletes = append(fixture.table().clone(), S6, vec![deletes]).await;
        let manifests = list(&with_deletes).await;
        assert_eq!(manifests[3].content, ManifestContentType::Deletes);
        let fx = Fixture {
            base: ExactTableBase::new(with_deletes, ()),
            mx: fixture.mx.clone(),
            mf: fixture.mf.clone(),
            mg: fixture.mg.clone(),
        };
        assert_refused(
            &fx,
            &view_of(S6, &manifests),
            vec![input(&fx.mf, data("f", 16), 4, 4)],
            vec![data("h", 64)],
            ErrorKind::FeatureUnsupported,
        )
        .await;

        // Format versions other than 2.
        for format_version in [FormatVersion::V1, FormatVersion::V3] {
            let other = Fixture {
                base: ExactTableBase::new(empty_table(format_version), ()),
                mx: fixture.mx.clone(),
                mf: fixture.mf.clone(),
                mg: fixture.mg.clone(),
            };
            assert_refused(
                &other,
                &other.view(),
                vec![input(&other.mf, data("f", 16), 4, 4)],
                vec![data("h", 64)],
                ErrorKind::FeatureUnsupported,
            )
            .await;
        }

        // Input or output under a spec that is not the default.
        let mut old_spec = data("o", 1);
        old_spec.partition_spec_id = 1;
        assert_refused(
            &fixture,
            &fixture.view(),
            vec![input(&fixture.mf, old_spec.clone(), 4, 4)],
            vec![data("h", 64)],
            ErrorKind::FeatureUnsupported,
        )
        .await;
        assert_refused(
            &fixture,
            &fixture.view(),
            vec![f()],
            vec![old_spec],
            ErrorKind::FeatureUnsupported,
        )
        .await;
        // An output whose partition does not fit the default spec.
        let mut misfit = data("m", 1);
        misfit.partition = Struct::from_iter([Some(Literal::string("x"))]);
        assert_refused(
            &fixture,
            &fixture.view(),
            vec![f()],
            vec![misfit],
            ErrorKind::DataInvalid,
        )
        .await;
    }

    #[tokio::test]
    async fn stage_checks_base() {
        let fixture = Fixture::new().await;
        let prepared = prepare_replace(
            &fixture.base,
            &fixture.view(),
            vec![input(&fixture.mf, data("f", 16), 4, 4)],
            vec![data("h", 64)],
        )
        .await
        .unwrap();
        let mut catalog = crate::catalog::MockCatalog::new();
        catalog.expect_stage_table_exact().times(0);

        // A base at another metadata location.
        let moved = Table::builder()
            .file_io(fixture.table().file_io().clone())
            .metadata(fixture.table().metadata_ref())
            .metadata_location(format!(
                "{}.moved",
                fixture.table().metadata_location().unwrap()
            ))
            .identifier(fixture.table().identifier().clone())
            .build()
            .unwrap();
        let error = prepared
            .stage(&catalog, ExactTableBase::new(moved, ()))
            .await
            .unwrap_err();
        assert!(
            matches!(error, ExactCommitError::BeforeCas { .. }),
            "{error}"
        );
    }
}
