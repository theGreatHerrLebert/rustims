//! What the failure that motivated this redesign looks like now.
use arrow::array::{Float64Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use timsim_schema::{tables::peptide_quantities as pq, write};

fn main() {
    // A stage names its output column after the model that produced it. This is exactly how
    // `retention_time_gru_predictor` happened.
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(pq::PEPTIDE_ID, DataType::UInt64, false),
            Field::new(pq::SAMPLE_ID, DataType::Utf8, false),
            Field::new("amount_amol_gru_predictor", DataType::Float64, false),
        ])),
        vec![
            Arc::new(UInt64Array::from(vec![1u64])),
            Arc::new(StringArray::from(vec!["A_R1"])),
            Arc::new(Float64Array::from(vec![1.0])),
        ],
    )
    .unwrap();

    match write("/tmp/x.parquet", "peptide_quantities", &batch, "timsim-yield/2.0", None) {
        Ok(()) => println!("written (this should not happen)"),
        Err(e) => println!("{e}"),
    }
}
