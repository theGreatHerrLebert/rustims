//! Shared plumbing for the protocol tools.
//!
//! Every tool obeys the same contract, because the CLI *is* the schema contract as far as
//! necroflow is concerned:
//!
//! ```text
//!   --out-*        explicit output paths (necroflow derives them; you never choose)
//!   --schema       print the output schema and exit
//!   --explain      print derived physical parameters and exit
//!   --report FILE  measured accounting as TOML — an error bound is data, not a log line
//!   --threads N    MUST NOT change the output
//! ```

use anyhow::Result;
use arrow::array::ArrayRef;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

pub mod spec;

/// Build a record batch against a registered table's schema, so the column order and types
/// come from the schema rather than from the caller's memory.
pub fn batch(table: &str, columns: Vec<ArrayRef>) -> Result<RecordBatch> {
    let spec = timsim_schema::tables::by_name(table)
        .ok_or_else(|| anyhow::anyhow!("unknown table {table:?}"))?;
    Ok(RecordBatch::try_new(spec.schema.clone(), columns)?)
}

/// Print a table's schema in a form a human can read and a machine can diff.
pub fn print_schema(table: &str) -> Result<()> {
    let spec = timsim_schema::tables::by_name(table)
        .ok_or_else(|| anyhow::anyhow!("unknown table {table:?}"))?;
    println!("table  : {}", spec.name);
    println!("axis   : {}", spec.axis.as_str());
    println!("version: {}", timsim_schema::SCHEMA_VERSION);
    println!();
    for f in spec.schema.fields() {
        println!(
            "  {:<22} {:<10} {}",
            f.name(),
            format!("{}", f.data_type()),
            if f.is_nullable() { "nullable" } else { "required" }
        );
    }
    Ok(())
}

/// The producer string stamped into every artifact, so an output knows what made it.
pub fn producer(tool: &str) -> String {
    format!("{tool}/{}", env!("CARGO_PKG_VERSION"))
}

/// Configure rayon. Thread count must never change the output — every tool is either analytic
/// or identity-seeded, so this is a performance knob and nothing else.
pub fn set_threads(n: Option<usize>) {
    if let Some(n) = n {
        let _ = rayon_threads(n);
    }
}

fn rayon_threads(n: usize) -> Result<()> {
    // timsim-chem uses rayon's global pool; setting it here keeps the tools honest about
    // --threads being purely a performance control.
    std::env::set_var("RAYON_NUM_THREADS", n.to_string());
    Ok(())
}

/// An empty schema, for the rare batch with no rows.
pub fn empty_schema() -> Arc<Schema> {
    Arc::new(Schema::empty())
}
