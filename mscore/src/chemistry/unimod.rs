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
    // R1 fold: modification compositions come from ms-chem (single source, cross-checked
    // against the mass table there).
    ms_chem::modification::MODIFICATION_COMPOSITION
        .iter()
        .map(|&(id, comp)| (format!("[UNIMOD:{id}]"), comp.iter().copied().collect()))
        .collect()
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