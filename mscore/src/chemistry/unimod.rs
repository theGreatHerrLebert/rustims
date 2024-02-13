use std::collections::HashMap;

pub fn modification_composition() -> HashMap<&'static str, HashMap<char, i32>> {
    let mut composition = HashMap::new();

    composition.insert("[UNIMOD:1]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Acetyl
    composition.insert("[UNIMOD:3]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Deamidated
    composition.insert("[UNIMOD:4]", HashMap::from([('C', 2), ('H', 3), ('O', 1)])); // Carbamidomethyl
    composition.insert("[UNIMOD:7]", HashMap::from([('H', 1)])); // Hydroxylation
    composition.insert("[UNIMOD:21]", HashMap::from([('O', 1)])); // Oxidation
    composition.insert("[UNIMOD:34]", HashMap::from([('C', 1), ('H', 1), ('O', 1)])); // Dehydrated
    composition.insert("[UNIMOD:35]", HashMap::from([('O', 1)])); // Phosphorylation
    composition.insert("[UNIMOD:36]", HashMap::from([('C', 2), ('H', 3), ('O', 1)])); // Pyroglutamic Acid
    composition.insert("[UNIMOD:37]", HashMap::from([('C', 2), ('H', 3), ('O', 1)])); // Pyroglutamic Acid
    composition.insert("[UNIMOD:43]", HashMap::from([('C', 7), ('H', 10), ('N', 1), ('O', 2)])); // Carboxymethyl
    composition.insert("[UNIMOD:58]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Dimethyl
    composition.insert("[UNIMOD:121]", HashMap::from([('C', 3), ('H', 4), ('O', 1)])); // Methylthio
    composition.insert("[UNIMOD:122]", HashMap::from([('C', 3), ('H', 4), ('O', 1)])); // Methylthio
    composition.insert("[UNIMOD:312]", HashMap::from([('C', 3), ('H', 4), ('O', 1)])); // Methylthio
    composition.insert("[UNIMOD:354]", HashMap::from([('C', 2), ('H', 4), ('O', 1)])); // Formyl
    composition.insert("[UNIMOD:408]", HashMap::from([('C', 5), ('H', 8), ('N', 1), ('O', 2)])); // Glutathione
    composition.insert("[UNIMOD:747]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Dimethyl:2H(6)
    composition.insert("[UNIMOD:1289]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Dimethyl:2H(6)
    composition.insert("[UNIMOD:1363]", HashMap::from([('C', 2), ('H', 2), ('O', 1)])); // Dimethyl:2H(6)

    composition
}

// MODIFICATIONS_MZ with string keys and float values
pub fn unimod_modifications_mz() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();
    map.insert("[UNIMOD:58]", 56.026215);
    map.insert("[UNIMOD:408]", 148.037173);
    map.insert("[UNIMOD:43]", 203.079373);
    map.insert("[UNIMOD:7]", 0.984016);
    map.insert("[UNIMOD:1]", 42.010565);
    map.insert("[UNIMOD:35]", 15.994915);
    map.insert("[UNIMOD:1289]", 70.041865);
    map.insert("[UNIMOD:3]", 226.077598);
    map.insert("[UNIMOD:1363]", 68.026215);
    map.insert("[UNIMOD:36]", 28.031300);
    map.insert("[UNIMOD:122]", 27.994915);
    map.insert("[UNIMOD:1848]", 114.031694);
    map.insert("[UNIMOD:1849]", 86.036779);
    map.insert("[UNIMOD:64]", 100.016044);
    map.insert("[UNIMOD:37]", 42.046950);
    map.insert("[UNIMOD:121]", 114.042927);
    map.insert("[UNIMOD:747]", 86.000394);
    map.insert("[UNIMOD:34]", 14.015650);
    map.insert("[UNIMOD:354]", 44.985078);
    map.insert("[UNIMOD:4]", 57.021464);
    map.insert("[UNIMOD:21]", 79.966331);
    map.insert("[UNIMOD:312]", 119.004099);
    map
}

// MODIFICATIONS_MZ_NUMERICAL with integer keys and float values
pub fn unimod_modifications_mz_numerical() -> HashMap<u32, f64> {
    let mut map = HashMap::new();
    map.insert(58, 56.026215);
    map.insert(408, 148.037173);
    map.insert(43, 203.079373);
    map.insert(7, 0.984016);
    map.insert(1, 42.010565);
    map.insert(35, 15.994915);
    map.insert(1289, 70.041865);
    map.insert(3, 226.077598);
    map.insert(1363, 68.026215);
    map.insert(36, 28.031300);
    map.insert(122, 27.994915);
    map.insert(1848, 114.031694);
    map.insert(1849, 86.036779);
    map.insert(64, 100.016044);
    map.insert(37, 42.046950);
    map.insert(121, 114.042927);
    map.insert(747, 86.000394);
    map.insert(34, 14.015650);
    map.insert(354, 44.985078);
    map.insert(4, 57.021464);
    map.insert(21, 79.966331);
    map.insert(312, 119.004099);
    map
}