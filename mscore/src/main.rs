use mscore::data::smiles::parse_peptide;

fn main() {
    let input = "[]-AC[U:4]DE[UNIMOD:35]F-[U:7]";
    match parse_peptide(input) {
        Ok(peptide) => println!("{:#?}", peptide),
        Err(e) => println!("Error: {}", e),
    }
}
