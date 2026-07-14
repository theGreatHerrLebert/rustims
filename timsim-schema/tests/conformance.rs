//! The tests that matter here are not "does Parquet round-trip" — they are "does a renamed
//! column get caught **at the stage boundary**".

use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use timsim_schema::{read, tables, validate, write, SchemaError};

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("timsim_schema_test_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d.join(name)
}

fn good_peptide_quantities() -> RecordBatch {
    let spec = tables::peptide_quantities::spec();
    RecordBatch::try_new(
        spec.schema.clone(),
        vec![
            Arc::new(UInt64Array::from(vec![1u64, 2, 3])),
            Arc::new(StringArray::from(vec!["A_R1", "A_R1", "B_R1"])),
            Arc::new(Float64Array::from(vec![100.0, 250.5, 12.25])),
        ],
    )
    .unwrap()
}

#[test]
fn round_trips_and_stamps_metadata() {
    let path = tmp("rt.parquet");
    write(&path, "peptide_quantities", &good_peptide_quantities(), "test/1.0", None).unwrap();

    let batches = read(&path, "peptide_quantities").unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);

    let msg = validate(&path, None).unwrap();
    assert!(msg.contains("peptide_quantities"), "{msg}");
    assert!(msg.contains("quantity"), "axis must be recorded: {msg}");
}

/// **THE test.** A column renamed after the model that produced it — the exact shape of
/// `retention_time_gru_predictor` — must fail at the boundary, naming the column, and not
/// three stages downstream inside a `row.get()`.
#[test]
fn a_renamed_column_is_caught_at_the_boundary_and_named() {
    let path = tmp("gru.parquet");

    // Someone helpfully names the amount column after the tool that made it.
    let bad_schema = Arc::new(Schema::new(vec![
        Field::new(tables::peptide_quantities::PEPTIDE_ID, DataType::UInt64, false),
        Field::new(tables::peptide_quantities::SAMPLE_ID, DataType::Utf8, false),
        Field::new("amount_amol_gru_predictor", DataType::Float64, false), // ← the disease
    ]));
    let bad = RecordBatch::try_new(
        bad_schema,
        vec![
            Arc::new(UInt64Array::from(vec![1u64])),
            Arc::new(StringArray::from(vec!["A_R1"])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    // It cannot even be written.
    let err = write(&path, "peptide_quantities", &bad, "test/1.0", None).unwrap_err();
    let msg = err.to_string();
    assert!(matches!(err, SchemaError::Nonconforming { .. }));
    assert!(
        msg.contains("amount_amol"),
        "the error must name the MISSING column: {msg}"
    );
    assert!(
        msg.contains("amount_amol_gru_predictor"),
        "and surface the suspicious extra column, since it is probably the renamed one: {msg}"
    );
}

/// Wiring the wrong artifact into a stage fails immediately, with both names.
#[test]
fn the_wrong_table_is_refused() {
    let path = tmp("wrong.parquet");
    write(&path, "peptide_quantities", &good_peptide_quantities(), "test/1.0", None).unwrap();

    let err = read(&path, "protein_quantities").unwrap_err();
    assert!(matches!(err, SchemaError::WrongTable { .. }));
    let msg = err.to_string();
    assert!(msg.contains("peptide_quantities") && msg.contains("protein_quantities"), "{msg}");
}

/// A retyped column is as dangerous as a renamed one — f32 where f64 was declared silently
/// truncates an attomole amount.
#[test]
fn a_retyped_column_is_caught() {
    let bad_schema = Arc::new(Schema::new(vec![
        Field::new(tables::peptide_quantities::PEPTIDE_ID, DataType::UInt64, false),
        Field::new(tables::peptide_quantities::SAMPLE_ID, DataType::Utf8, false),
        Field::new(tables::peptide_quantities::AMOUNT_AMOL, DataType::Float32, false),
    ]));
    let bad = RecordBatch::try_new(
        bad_schema,
        vec![
            Arc::new(UInt64Array::from(vec![1u64])),
            Arc::new(StringArray::from(vec!["A_R1"])),
            Arc::new(arrow::array::Float32Array::from(vec![1.0f32])),
        ],
    )
    .unwrap();

    let err = write(tmp("retyped.parquet"), "peptide_quantities", &bad, "test/1.0", None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Float32") && msg.contains("Float64"), "{msg}");
}

/// Extra columns are allowed — a stage may annotate without breaking its consumers.
#[test]
fn extra_columns_are_permitted() {
    let path = tmp("extra.parquet");
    let spec = tables::peptide_quantities::spec();
    let mut fields: Vec<Field> = spec.schema.fields().iter().map(|f| (**f).clone()).collect();
    fields.push(Field::new("my_debug_note", DataType::Utf8, true));

    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        vec![
            Arc::new(UInt64Array::from(vec![1u64])),
            Arc::new(StringArray::from(vec!["A_R1"])),
            Arc::new(Float64Array::from(vec![1.0])),
            Arc::new(StringArray::from(vec!["hello"])),
        ],
    )
    .unwrap();

    write(&path, "peptide_quantities", &batch, "test/1.0", None).unwrap();
    assert_eq!(read(&path, "peptide_quantities").unwrap()[0].num_columns(), 4);
}

/// `peptide_occurrences` must NOT carry a yield column. Yield depends on digestion efficiency
/// and on regulated cleavage-blocking modifications, so it is condition-dependent — putting it
/// on a structure table would silently destroy sharing across samples.
#[test]
fn the_structure_axis_carries_no_quantities() {
    let occ = tables::peptide_occurrences::spec();
    assert_eq!(occ.axis, timsim_schema::Axis::Structure);
    for f in occ.schema.fields() {
        let n = f.name();
        assert!(
            !n.contains("yield") && !n.contains("amol") && !n.contains("amount"),
            "structure table {:?} must not carry the quantity column {n:?}",
            occ.name
        );
    }

    let prot = tables::proteome::spec();
    for f in prot.schema.fields() {
        assert!(
            !f.name().contains("amount") && !f.name().contains("abundance"),
            "the proteome is structure: amounts live in protein_quantities"
        );
    }
}

/// Every table is reachable by name from the registry, and every name is unique.
#[test]
fn the_registry_is_complete_and_unambiguous() {
    let all = tables::all();
    assert!(all.len() >= 9);
    for t in &all {
        assert!(tables::by_name(t.name).is_some(), "{} not resolvable", t.name);
    }
    let mut names: Vec<&str> = all.iter().map(|t| t.name).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "duplicate table name in the registry");
}

/// REGRESSION: a required column arriving as nullable must be refused.
///
/// The conformance check originally compared only names and types. A nullable `amount_amol` was
/// accepted, and downstream readers call `.value(i)` on it — which on a null returns **garbage
/// rather than failing**. A silently wrong attomole amount is worse than a crash. Found by review.
#[test]
fn a_required_column_arriving_nullable_is_refused() {
    let bad_schema = Arc::new(Schema::new(vec![
        Field::new(tables::peptide_quantities::PEPTIDE_ID, DataType::UInt64, false),
        Field::new(tables::peptide_quantities::SAMPLE_ID, DataType::Utf8, false),
        Field::new(tables::peptide_quantities::AMOUNT_AMOL, DataType::Float64, true), // ← nullable
    ]));
    let bad = RecordBatch::try_new(
        bad_schema,
        vec![
            Arc::new(UInt64Array::from(vec![1u64])),
            Arc::new(StringArray::from(vec!["A_R1"])),
            Arc::new(Float64Array::from(vec![Some(1.0)])),
        ],
    )
    .unwrap();

    let err = write(tmp("nullable.parquet"), "peptide_quantities", &bad, "test/1.0", None)
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("nullable"), "{msg}");
    assert!(msg.contains("amount_amol"), "{msg}");
}

/// REGRESSION: an unstamped Parquet file must not be accepted as a timsim artifact.
///
/// The identity and version checks were `if let Some(...)`, so a file with NO metadata skipped
/// both and was accepted purely because its columns lined up. That defeats the whole point of a
/// stage-boundary contract: a generic Parquet file — or a legacy artifact from before the schema
/// existed — could be wired into a stage without ever proving what it was. Found by review.
#[test]
fn an_unstamped_parquet_file_is_not_a_timsim_artifact() {
    let path = tmp("unstamped.parquet");

    // Right columns, right types — but written by something that is not us.
    let batch = good_peptide_quantities();
    let f = std::fs::File::create(&path).unwrap();
    let mut w = parquet::arrow::ArrowWriter::try_new(f, batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let err = read(&path, "peptide_quantities").unwrap_err();
    assert!(matches!(err, SchemaError::Unstamped { .. }), "got: {err}");
    assert!(err.to_string().contains("not a timsim artifact"), "{err}");
}
