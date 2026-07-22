use std::collections::HashMap;

/// Unimod Modifications
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<String, HashMap<&'static str, i32>>` - a map of unimod modification names to their atomic compositions
///
/// # Example
///
/// ```
/// use mscore::chemistry::unimod::modification_atomic_composition;
/// use std::collections::HashMap;
///
/// let composition = modification_atomic_composition();
/// assert_eq!(composition.get("[UNIMOD:1]"), Some(&HashMap::from([("C", 2), ("H", 2), ("O", 1)])));
/// ```
pub fn modification_atomic_composition() -> HashMap<String, HashMap<&'static str, i32>> {
    let mut composition: HashMap<String, HashMap<&'static str, i32>> = HashMap::new();
    composition.insert("[UNIMOD:1]".to_string(), HashMap::from([("C", 2), ("H", 2), ("O", 1)])); // Acetyl
    composition.insert("[UNIMOD:3]".to_string(), HashMap::from([("N", 2), ("C", 10), ("H", 14), ("O", 2), ("S", 1)])); //  	Biotinylation
    composition.insert("[UNIMOD:4]".to_string(), HashMap::from([("C", 2), ("H", 3), ("O", 1), ("N", 1)]));
    composition.insert("[UNIMOD:7]".to_string(), HashMap::from([("H", -1), ("N", -1), ("O", 1)])); // Hydroxylation
    composition.insert("[UNIMOD:21]".to_string(), HashMap::from([("H", 1),("O", 3), ("P", 1)])); // Phosphorylation
    composition.insert("[UNIMOD:34]".to_string(), HashMap::from([("H", 2), ("C", 1)])); //  Methylation
    composition.insert("[UNIMOD:35]".to_string(), HashMap::from([("O", 1)])); // Hydroxylation
    // composition.insert("[UNIMOD:43]".to_string(), HashMap::from([("C", 8), ("H", 15), ("N", 1), ("O", 6)])); // HexNAc ??
    composition.insert("[UNIMOD:58]".to_string(), HashMap::from([("C", 3), ("H", 4), ("O", 1)])); // Propionyl
    composition.insert("[UNIMOD:121]".to_string(), HashMap::from([("C", 4), ("H", 6), ("O", 2), ("N", 2)])); // ubiquitinylation residue
    composition.insert("[UNIMOD:122]".to_string(), HashMap::from([("C", 1), ("O", 1)])); // Formylation
    composition.insert("[UNIMOD:312]".to_string(), HashMap::from([("C", 3), ("H", 5), ("O", 2), ("N", 1), ("S", 1)])); // Cysteinyl
    composition.insert("[UNIMOD:354]".to_string(), HashMap::from([("H", -1), ("O", 2), ("N", 1)])); // Oxidation to nitro
    // composition.insert("[UNIMOD:408]".to_string(), HashMap::from([("C", -1), ("H", -2), ("N", 1), ("O", 2)])); // Glycosyl ??
    composition.insert("[UNIMOD:747]".to_string(), HashMap::from([("C", 3), ("H", 2), ("O", 3)])); // Malonylation
    composition.insert("[UNIMOD:1289]".to_string(), HashMap::from([("C", 4), ("H", 6), ("O", 1)])); // Butyryl
    composition.insert("[UNIMOD:1363]".to_string(), HashMap::from([("C", 4), ("H", 4), ("O", 1)])); // Crotonylation

    composition
}

/// Unimod Modifications Mass
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, f64>` - a map of unimod modification names to their mass
///
/// # Example
///
/// ```
/// use mscore::chemistry::unimod::unimod_modifications_mass;
///
/// let mass = unimod_modifications_mass();
/// assert_eq!(mass.get("[UNIMOD:1]"), Some(&42.010565));
/// ```
pub fn unimod_modifications_mass() -> HashMap<String, f64> {
    // R1 fold: derived from ms-chem's numerical table (keys formatted from ids).
    ms_chem::unimod::UNIMOD_MASS
        .iter()
        .map(|&(id, m)| (format!("[UNIMOD:{id}]"), m))
        .collect()
}

/// Unimod Modifications Mass Numerical
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<u32, f64>` - a map of unimod modification numerical ids to their mass
///
/// # Example
///
/// ```
/// use mscore::chemistry::unimod::unimod_modifications_mass_numerical;
///
/// let mass = unimod_modifications_mass_numerical();
/// assert_eq!(mass.get(&1), Some(&42.010565));
/// ```
pub fn unimod_modifications_mass_numerical() -> HashMap<u32, f64> {
    // R1 fold: the full UNIMOD mass table lives in ms-chem now (single source).
    ms_chem::unimod::UNIMOD_MASS.iter().copied().collect()
}