use std::collections::HashMap;

/// Amino Acids
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, &'static str>` - a map of amino acid names to their one-letter codes
///
/// # Example
///
/// ```
/// use mscore::chemistry::amino_acid::amino_acids;
///
/// let amino_acids = amino_acids();
/// assert_eq!(amino_acids.get("Lysine"), Some(&"K"));
/// ```
pub fn amino_acids() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    map.insert("Lysine", "K");
    map.insert("Alanine", "A");
    map.insert("Glycine", "G");
    map.insert("Valine", "V");
    map.insert("Tyrosine", "Y");
    map.insert("Arginine", "R");
    map.insert("Glutamic Acid", "E");
    map.insert("Phenylalanine", "F");
    map.insert("Tryptophan", "W");
    map.insert("Leucine", "L");
    map.insert("Threonine", "T");
    map.insert("Cysteine", "C");
    map.insert("Serine", "S");
    map.insert("Glutamine", "Q");
    map.insert("Methionine", "M");
    map.insert("Isoleucine", "I");
    map.insert("Asparagine", "N");
    map.insert("Proline", "P");
    map.insert("Histidine", "H");
    map.insert("Aspartic Acid", "D");
    map.insert("Selenocysteine", "U");
    map
}


/// Amino Acid Masses
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, f64>` - a map of amino acid one-letter codes to their monoisotopic masses
///
/// # Example
///
/// ```
/// use mscore::chemistry::amino_acid::amino_acid_masses;
///
/// let amino_acid_masses = amino_acid_masses();
/// assert_eq!(amino_acid_masses.get("K"), Some(&128.094963));
/// ```
pub fn amino_acid_masses() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();
    map.insert("A", 71.037114);
    map.insert("R", 156.101111);
    map.insert("N", 114.042927);
    map.insert("D", 115.026943);
    map.insert("C", 103.009185);
    map.insert("E", 129.042593);
    map.insert("Q", 128.058578);
    map.insert("G", 57.021464);
    map.insert("H", 137.058912);
    map.insert("I", 113.084064);
    map.insert("L", 113.084064);
    map.insert("K", 128.094963);
    map.insert("M", 131.040485);
    map.insert("F", 147.068414);
    map.insert("P", 97.052764);
    map.insert("S", 87.032028);
    map.insert("T", 101.047679);
    map.insert("W", 186.079313);
    map.insert("Y", 163.063329);
    map.insert("V", 99.068414);
    map.insert("U", 168.053);
    map
}

/// Amino Acid Composition
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<char, HashMap<&'static str, i32>>` - a map of amino acid one-letter codes to their atomic compositions
///
/// # Example
///
/// ```
/// use mscore::chemistry::amino_acid::amino_acid_composition;
/// use std::collections::HashMap;
///
/// let amino_acid_composition = amino_acid_composition();
/// assert_eq!(amino_acid_composition.get(&'K'), Some(&HashMap::from([("C", 6), ("H", 12), ("N", 2), ("O", 1)])));
/// ```
pub fn amino_acid_composition() -> HashMap<char, HashMap<&'static str, i32>> {

    let mut composition: HashMap<char, HashMap<&'static str, i32>> = HashMap::new();

    composition.insert('G', HashMap::from([("C", 2), ("H", 3), ("N", 1), ("O", 1)])); // Glycine
    composition.insert('A', HashMap::from([("C", 3), ("H", 5), ("N", 1), ("O", 1)])); // Alanine
    composition.insert('S', HashMap::from([("C", 3), ("H", 5), ("N", 1), ("O", 2)])); // Serine
    composition.insert('P', HashMap::from([("C", 5), ("H", 7), ("N", 1), ("O", 1)])); // Proline
    composition.insert('V', HashMap::from([("C", 5), ("H", 9), ("N", 1), ("O", 1)])); // Valine
    composition.insert('T', HashMap::from([("C", 4), ("H", 7), ("N", 1), ("O", 2)])); // Threonine
    composition.insert('C', HashMap::from([("C", 3), ("H", 5), ("N", 1), ("O", 1), ("S", 1)])); // Cysteine
    composition.insert('I', HashMap::from([("C", 6), ("H", 11), ("N", 1), ("O", 1)])); // Isoleucine
    composition.insert('L', HashMap::from([("C", 6), ("H", 11), ("N", 1), ("O", 1)])); // Leucine
    composition.insert('N', HashMap::from([("C", 4), ("H", 6), ("N", 2), ("O", 2)])); // Asparagine
    composition.insert('D', HashMap::from([("C", 4), ("H", 5), ("N", 1), ("O", 3)])); // Aspartic Acid
    composition.insert('Q', HashMap::from([("C", 5), ("H", 8), ("N", 2), ("O", 2)])); // Glutamine
    composition.insert('K', HashMap::from([("C", 6), ("H", 12), ("N", 2), ("O", 1)])); // Lysine
    composition.insert('E', HashMap::from([("C", 5), ("H", 7), ("N", 1), ("O", 3)])); // Glutamic Acid
    composition.insert('M', HashMap::from([("C", 5), ("H", 9), ("N", 1), ("O", 1), ("S", 1)])); // Methionine
    composition.insert('H', HashMap::from([("C", 6), ("H", 7), ("N", 3), ("O", 1)])); // Histidine
    composition.insert('F', HashMap::from([("C", 9), ("H", 9), ("N", 1), ("O", 1)])); // Phenylalanine
    composition.insert('R', HashMap::from([("C", 6), ("H", 12), ("N", 4), ("O", 1)])); // Arginine
    composition.insert('Y', HashMap::from([("C", 9), ("H", 9), ("N", 1), ("O", 2)])); // Tyrosine
    composition.insert('W', HashMap::from([("C", 11), ("H", 10), ("N", 2), ("O", 1)])); // Tryptophan
    composition.insert('U', HashMap::from([("C", 3), ("H", 5), ("N", 1), ("O", 1), ("Se", 1)])); // Selenocysteine

    composition
}