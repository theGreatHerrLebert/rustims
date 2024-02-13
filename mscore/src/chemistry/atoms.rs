extern crate statrs;
use statrs::distribution::{Continuous, Normal, Binomial, Discrete};
use regex::Regex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use crate::{MzSpectrum, ToResolution};
use std::collections::HashMap;

fn normal_pdf(x: f64, mean: f64, std_dev: f64) -> f64 {
    let normal = Normal::new(mean, std_dev).unwrap();
    normal.pdf(x)
}

pub const MASS_PROTON: f64 = 1.007276466621;
pub const MASS_NEUTRON: f64 = 1.00866491595;
pub const MASS_ELECTRON: f64 = 0.00054857990946;
pub const MASS_WATER: f64 = 18.0105646863;

// IUPAC Standards
pub const STANDARD_TEMPERATURE: f64 = 273.15; // Kelvin
pub const STANDARD_PRESSURE: f64 = 1e5; // Pascal
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;
pub const K_BOLTZMANN: f64 = 1.380649e-23; // J/K

// Amino Acids and Their Codes
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

// Amino Acid Masses
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

pub fn amino_acid_composition() -> HashMap<char, HashMap<&'static str, i32>> {

    let mut composition: HashMap<char, HashMap<&'static str, i32>> = HashMap::new();

    composition.insert('G', HashMap::from([("C", 2), ("H", 3), ("N", 1), ("O", 2)])); // Glycine
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

pub fn atomic_weights_isotope_averaged() -> HashMap<&'static str, f64> {
    let mut map = HashMap::new();
    map.insert("H", 1.00782503223);
    map.insert("He", 4.00260325415);
    map.insert("Li", 7.0160034366);
    map.insert("Be", 9.012183065);
    map.insert("B", 11.00930536);
    map.insert("C", 12.0000000);
    map.insert("N", 14.00307400443);
    map.insert("O", 15.99491461957);
    map.insert("F", 18.99840316273);
    map.insert("Ne", 19.9924401762);
    map.insert("Na", 22.9897692820);
    map.insert("Mg", 23.985041697);
    map.insert("Al", 26.98153853);
    map.insert("Si", 27.97692653465);
    map.insert("P", 30.97376199842);
    map.insert("S", 31.9720711744);
    map.insert("Cl", 34.968852682);
    map.insert("Ar", 39.9623831237);
    map.insert("K", 38.963706679);
    map.insert("Ca", 39.96259098);
    map.insert("Sc", 44.95590828);
    map.insert("Ti", 47.9479463);
    map.insert("V", 50.9439595);
    map.insert("Cr", 51.9405075);
    map.insert("Mn", 54.9380455);
    map.insert("Fe", 55.9349375);
    map.insert("Co", 58.9331955);
    map.insert("Ni", 57.9353429);
    map.insert("Cu", 62.9295975);
    map.insert("Zn", 63.9291422);
    map.insert("Ga", 68.9255735);
    map.insert("Ge", 73.9211778);
    map.insert("As", 74.9215965);
    map.insert("Se", 79.9165218);
    map.insert("Br", 78.9183376);
    map.insert("Kr", 83.911507);
    map.insert("Rb", 84.9117893);
    map.insert("Sr", 87.9056125);
    map.insert("Y", 88.905842);
    map.insert("Zr", 89.9046977);
    map.insert("Nb", 92.906373);
    map.insert("Mo", 97.905404);
    map.insert("Tc", 98.0);
    map.insert("Ru", 101.904349);
    map.insert("Rh", 102.905504);
    map.insert("Pd", 105.903485);
    map.insert("Ag", 106.905093);
    map.insert("Cd", 113.903358);
    map.insert("In", 114.903878);
    map.insert("Sn", 119.902199);
    map.insert("Sb", 120.903818);
    map.insert("Te", 129.906224);
    map.insert("I", 126.904473);
    map.insert("Xe", 131.904155);
    map.insert("Cs", 132.905447);
    map.insert("Ba", 137.905247);
    map.insert("La", 138.906355);
    map.insert("Ce", 139.905442);
    map.insert("Pr", 140.907662);
    map.insert("Nd", 141.907732);
    map.insert("Pm", 145.0);
    map.insert("Sm", 151.919728);
    map.insert("Eu", 152.921225);
    map.insert("Gd", 157.924103);
    map.insert("Tb", 158.925346);
    map.insert("Dy", 163.929171);
    map.insert("Ho", 164.930319);
    map.insert("Er", 165.930290);
    map.insert("Tm", 168.934211);
    map.insert("Yb", 173.938859);
    map.insert("Lu", 174.940770);
    map.insert("Hf", 179.946550);
    map.insert("Ta", 180.947992);
    map.insert("W", 183.950932);
    map.insert("Re", 186.955744);
    map.insert("Os", 191.961467);
    map.insert("Ir", 192.962917);
    map.insert("Pt", 194.964766);
    map.insert("Au", 196.966543);
    map.insert("Hg", 201.970617);
    map.insert("Tl", 204.974427);
    map.insert("Pb", 207.976627);
    map.insert("Bi", 208.980384);
    map.insert("Po", 209.0);
    map.insert("At", 210.0);
    map.insert("Rn", 222.0);
    map.insert("Fr", 223.0);
    map.insert("Ra", 226.0);
    map.insert("Ac", 227.0);
    map.insert("Th", 232.038054);
    map.insert("Pa", 231.035882);
    map.insert("U", 238.050786);
    map.insert("Np", 237.0);
    map.insert("Pu", 244.0);
    map.insert("Am", 243.0);
    map.insert("Cm", 247.0);
    map.insert("Bk", 247.0);
    map.insert("Cf", 251.0);
    map.insert("Es", 252.0);
    map.insert("Fm", 257.0);
    map.insert("Md", 258.0);
    map.insert("No", 259.0);
    map.insert("Lr", 262.0);
    map.insert("Rf", 267.0);
    map.insert("Db", 270.0);
    map.insert("Sg", 271.0);
    map.insert("Bh", 270.0);
    map.insert("Hs", 277.0);
    map.insert("Mt", 276.0);
    map.insert("Ds", 281.0);
    map.insert("Rg", 280.0);
    map.insert("Cn", 285.0);
    map.insert("Nh", 284.0);
    map.insert("Fl", 289.0);
    map.insert("Mc", 288.0);
    map.insert("Lv", 293.0);
    map.insert("Ts", 294.0);
    map.insert("Og", 294.0);

    map
}

pub fn atoms_isotopic_weights() -> HashMap<&'static str, Vec<f64>> {
    let mut map = HashMap::new();
    map.insert("H", vec![1.00782503223, 2.01410177812]);
    map.insert("He", vec![4.00260325415]);
    map.insert("Li", vec![7.0160034366]);
    map.insert("Be", vec![9.012183065]);
    map.insert("B", vec![11.00930536]);
    map.insert("C", vec![12.0000000, 13.00335483507]);
    map.insert("N", vec![14.00307400443, 15.00010889888]);
    map.insert("O", vec![15.99491461957, 16.99913175650, 17.99915961286]);
    map.insert("F", vec![18.99840316273]);
    map.insert("Ne", vec![19.9924401762]);
    map.insert("Na", vec![22.9897692820]);
    map.insert("Mg", vec![23.985041697]);
    map.insert("Al", vec![26.98153853]);
    map.insert("Si", vec![27.97692653465]);
    map.insert("P", vec![30.97376199842]);
    map.insert("S", vec![31.9720711744, 32.9714589098, 33.967867004]);
    map.insert("Cl", vec![34.968852682, 36.965902602]);
    map.insert("Ar", vec![39.9623831237, 35.967545105]);
    map.insert("K", vec![38.963706679, 39.963998166, 40.961825257]);
    map.insert("Ca", vec![39.96259098, 41.95861783, 42.95876644, 43.95548156, 45.95369276]);
    map.insert("Sc", vec![44.95590828]);
    map.insert("Ti", vec![47.9479463, 45.95262772, 46.95175879, 47.94794198, 49.9447912]);
    map.insert("V", vec![50.9439595]);
    map.insert("Cr", vec![51.9405075, 49.94604183, 50.9439637, 51.94050623, 53.93887916]);
    map.insert("Mn", vec![54.9380455]);
    map.insert("Fe", vec![55.9349375, 53.93960899, 54.93804514, 55.93493739, 56.93539400, 57.93327443]);
    map.insert("Co", vec![58.9331955]);
    map.insert("Ni", vec![57.9353429, 58.9343467, 59.93078588, 60.93105557, 61.92834537, 63.92796682]);
    map.insert("Cu", vec![62.9295975, 61.92834537, 63.92914201]);
    map.insert("Zn", vec![63.9291422, 65.92603381, 66.92712775, 67.92484455, 69.9253192]);
    map.insert("Ga", vec![68.9255735]);
    map.insert("Ge", vec![73.9211778, 71.922075826, 72.923458956, 73.921177761, 75.921402726]);
    map.insert("As", vec![74.9215965]);
    map.insert("Se", vec![79.9165218, 73.9224764, 75.9192136, 76.9199140, 77.9173095, 79.9165218, 81.9166995]);

    map
}

pub fn isotopic_abundance() -> HashMap<&'static str, Vec<f64>> {

    let mut map = HashMap::new();

    map.insert("H", vec![0.999885, 0.000115]);
    map.insert("He", vec![0.99999866, 0.00000134]);
    map.insert("Li", vec![0.0759, 0.9241]);
    map.insert("Be", vec![1.0]);
    map.insert("B", vec![0.199, 0.801]);
    map.insert("C", vec![0.9893, 0.0107]);
    map.insert("N", vec![0.99632, 0.00368]);
    map.insert("O", vec![0.99757, 0.00038, 0.00205]);
    map.insert("F", vec![1.0]);
    map.insert("Ne", vec![0.9048, 0.0027, 0.0925]);
    map.insert("Na", vec![0.5429, 0.4571]);
    map.insert("Mg", vec![0.7899, 0.1000, 0.1101]);
    map.insert("Al", vec![1.0]);
    map.insert("Si", vec![0.9223, 0.0467, 0.0310]);
    map.insert("P", vec![1.0]);
    map.insert("S", vec![0.9493, 0.0076, 0.0429]);
    map.insert("Cl", vec![0.7578, 0.2422]);
    map.insert("Ar", vec![0.003365, 0.000632, 0.996003]);
    map.insert("K", vec![0.932581, 0.000117, 0.067302]);
    map.insert("Ca", vec![0.96941, 0.00647, 0.00135, 0.02086, 0.00187]);
    map.insert("Sc", vec![1.0]);
    map.insert("Ti", vec![0.0825, 0.0744, 0.7372, 0.0541, 0.0518]);
    map.insert("V", vec![0.9975, 0.0025]);
    map.insert("Cr", vec![0.04345, 0.83789, 0.09501, 0.02365, 0.0001]);
    map.insert("Mn", vec![1.0]);
    map.insert("Fe", vec![0.05845, 0.91754, 0.02119, 0.00282, 0.0002]);
    map.insert("Co", vec![1.0]);
    map.insert("Ni", vec![0.680769, 0.262231, 0.011399, 0.036345, 0.009256, 0.0011]);
    map.insert("Cu", vec![0.6915, 0.3085]);
    map.insert("Zn", vec![0.4917, 0.2773, 0.0404, 0.1845, 0.0061]);
    map.insert("Ga", vec![0.60108, 0.39892]);
    map.insert("Ge", vec![0.2052, 0.2745, 0.0775, 0.3652, 0.0775]);
    map.insert("As", vec![1.0]);
    map.insert("Se", vec![0.0089, 0.0937, 0.0763, 0.2377, 0.4961, 0.0873]);

    map
}


// MODIFICATIONS_MZ with string keys and float values
pub fn modifications_mz() -> HashMap<&'static str, f64> {
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
pub fn modifications_mz_numerical() -> HashMap<u32, f64> {
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

/// calculate the monoisotopic mass of a peptide sequence
///
/// Arguments:
///
/// * `sequence` - peptide sequence
///
/// Returns:
///
/// * `mass` - monoisotopic mass of the peptide
///
/// # Examples
///
/// ```
/// use mscore::calculate_monoisotopic_mass;
///
/// let mass = calculate_monoisotopic_mass("PEPTIDEC[UNIMOD:4]R");
/// // assert_eq!(mass, 1115.4917246863);
/// ```
pub fn calculate_monoisotopic_mass(sequence: &str) -> f64 {
    let amino_acid_masses = amino_acid_masses();
    let modifications_mz_numerical = modifications_mz_numerical();
    let pattern = Regex::new(r"\[UNIMOD:(\d+)\]").unwrap();

    // Find all occurrences of the pattern
    let modifications: Vec<u32> = pattern
        .find_iter(sequence)
        .filter_map(|mat| mat.as_str()[8..mat.as_str().len() - 1].parse().ok())
        .collect();

    // Remove the modifications from the sequence
    let sequence = pattern.replace_all(sequence, "");

    // Count occurrences of each amino acid
    let mut aa_counts = HashMap::new();
    for char in sequence.chars() {
        *aa_counts.entry(char).or_insert(0) += 1;
    }

    // Mass of amino acids and modifications
    let mass_sequence: f64 = aa_counts.iter().map(|(aa, &count)| amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0) * count as f64).sum();
    let mass_modifics: f64 = modifications.iter().map(|&mod_id| modifications_mz_numerical.get(&mod_id).unwrap_or(&0.0)).sum();

    mass_sequence + mass_modifics + MASS_WATER
}

pub fn calculate_b_y_fragment_mz(sequence: &str, modifications: Vec<f64>, is_y: Option<bool>, charge: Option<i32>) -> f64 {
    // Return mz of empty sequence
    if sequence.is_empty() {
        return 0.0;
    }

    let amino_acid_masses = amino_acid_masses();

    // Add up raw amino acid masses and potential modifications
    let mass_sequence: f64 = sequence.chars()
        .map(|aa| amino_acid_masses.get(&aa.to_string()[..]).unwrap_or(&0.0))
        .sum();

    let mass_modifications: f64 = modifications.iter().sum();

    // Calculate total mass
    let mass = mass_sequence + mass_modifications;

    // Set default values if None
    let is_y = is_y.unwrap_or(false);
    let charge = charge.unwrap_or(1);

    // If sequence is n-terminal (is_y is true), add water mass and calculate mz
    if is_y {
        calculate_mz(mass + MASS_WATER, charge)
    } else {
        // Otherwise, calculate mz
        calculate_mz(mass, charge)
    }
}

pub fn calculate_b_y_ion_series(sequence: &str, modifications: Vec<f64>, charge: Option<i32>) -> (Vec<(f64, String, String)>, Vec<(f64, String, String)>) {
    let mut b_ions = Vec::new();
    let mut y_ions = Vec::new();

    let char_indices: Vec<usize> = sequence.char_indices().map(|(i, _)| i).collect();
    let sequence_length = char_indices.len();

    // Iterate over all possible cleavage sites
    for i in 0..=sequence_length {
        let b_index = *char_indices.get(i).unwrap_or(&sequence.len());
        let y_index = *char_indices.get(i).unwrap_or(&0);

        let y = &sequence[y_index..];
        let b = &sequence[..b_index];
        let m_y = &modifications[i..];
        let m_b = &modifications[..i];

        // Calculate mz of b ions
        if !b.is_empty() && i != sequence_length {
            let b_mass = calculate_b_y_fragment_mz(b, m_b.to_vec(), Some(false), charge);
            b_ions.push((b_mass, format!("b{}+{}", i, charge.unwrap_or(1)), b.to_string()));
        }

        // Calculate mz of y ions
        if !y.is_empty() && i != 0 && i != sequence_length {
            let y_mass = calculate_b_y_fragment_mz(y, m_y.to_vec(), Some(true), charge);
            y_ions.push((y_mass, format!("y{}+{}", sequence_length - i, charge.unwrap_or(1)), y.to_string()));
        }
    }

    (b_ions, y_ions)
}


/// calculate the m/z of an ion
///
/// Arguments:
///
/// * `mono_mass` - monoisotopic mass of the ion
/// * `charge` - charge state of the ion
///
/// Returns:
///
/// * `mz` - mass-over-charge of the ion
///
/// # Examples
///
/// ```
/// use mscore::calculate_mz;
///
/// let mz = calculate_mz(1000.0, 2);
/// assert_eq!(mz, 501.007276466621);
/// ```
pub fn calculate_mz(monoisotopic_mass: f64, charge: i32) -> f64 {
    (monoisotopic_mass + charge as f64 * MASS_PROTON) / charge as f64
}

/// convert 1 over reduced ion mobility (1/k0) to CCS
///
/// Arguments:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `ccs` - collision cross-section
///
/// # Examples
///
/// ```
/// use mscore::one_over_reduced_mobility_to_ccs;
///
/// let ccs = one_over_reduced_mobility_to_ccs(0.5, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(ccs, 806.5918693771381);
/// ```
pub fn one_over_reduced_mobility_to_ccs(
    one_over_k0: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    summary_constant * charge as f64 / (reduced_mass * (temp + t_diff)).sqrt() / one_over_k0
}


/// convert CCS to 1 over reduced ion mobility (1/k0)
///
/// Arguments:
///
/// * `ccs` - collision cross-section
/// * `charge` - charge state of the ion
/// * `mz` - mass-over-charge of the ion
/// * `mass_gas` - mass of drift gas (N2)
/// * `temp` - temperature of the drift gas in C°
/// * `t_diff` - factor to translate from C° to K
///
/// Returns:
///
/// * `one_over_k0` - 1 over reduced ion mobility (1/k0)
///
/// # Examples
///
/// ```
/// use mscore::ccs_to_reduced_mobility;
///
/// let k0 = ccs_to_reduced_mobility(806.5918693771381, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(1.0 / k0, 0.5);
/// ```
pub fn ccs_to_reduced_mobility(
    ccs: f64,
    mz: f64,
    charge: u32,
    mass_gas: f64,
    temp: f64,
    t_diff: f64,
) -> f64 {
    let summary_constant = 18509.8632163405;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    ((reduced_mass * (temp + t_diff)).sqrt() * ccs) / (summary_constant * charge as f64)
}


/// calculate the factorial of a number
///
/// Arguments:
///
/// * `n` - number to calculate factorial of
///
/// Returns:
///
/// * `f64` - factorial of `n`
///
/// # Examples
///
/// ```
/// use mscore::factorial;
///
/// let fact = factorial(5);
/// assert_eq!(fact, 120.0);
/// ```
pub fn factorial(n: i32) -> f64 {
    (1..=n).fold(1.0, |acc, x| acc * x as f64)
}

fn weight(mass: f64, peak_nums: Vec<i32>, normalize: bool) -> Vec<f64> {
    let lam_val = lam(mass, 0.000594, -0.03091);
    let factorials: Vec<f64> = peak_nums.iter().map(|&k| factorial(k)).collect();
    let mut weights: Vec<f64> = peak_nums.iter().map(|&k| {
        let pow = lam_val.powi(k);
        let exp = (-lam_val).exp();
        exp * pow / factorials[k as usize]
    }).collect();

    if normalize {
        let sum: f64 = weights.iter().sum();
        weights = weights.iter().map(|&w| w / sum).collect();
    }

    weights
}

fn lam(mass: f64, slope: f64, intercept: f64) -> f64 {
    slope * mass + intercept
}

fn iso(x: &Vec<f64>, mass: f64, charge: f64, sigma: f64, amp: f64, k: usize, step_size: f64, mass_neutron: f64) -> Vec<f64> {
    let k_range: Vec<usize> = (0..k).collect();
    let means: Vec<f64> = k_range.iter().map(|&k_val| (mass + mass_neutron * k_val as f64) / charge).collect();
    let weights = weight(mass, k_range.iter().map(|&k_val| k_val as i32).collect::<Vec<i32>>(), true);

    let mut intensities = vec![0.0; x.len()];
    for (i, x_val) in x.iter().enumerate() {
        for (j, &mean) in means.iter().enumerate() {
            intensities[i] += weights[j] * normal_pdf(*x_val, mean, sigma);
        }
        intensities[i] *= step_size;
    }
    intensities.iter().map(|&intensity| intensity * amp).collect()
}

fn generate_isotope_pattern(lower_bound: f64, upper_bound: f64, mass: f64, charge: f64, amp: f64, k: usize, sigma: f64, resolution: i32) -> (Vec<f64>, Vec<f64>) {
    let step_size = f64::min(sigma / 10.0, 1.0 / 10f64.powi(resolution));
    let size = ((upper_bound - lower_bound) / step_size).ceil() as usize;
    let mzs: Vec<f64> = (0..size).map(|i| lower_bound + step_size * i as f64).collect();
    let intensities = iso(&mzs, mass, charge, sigma, amp, k, step_size, MASS_NEUTRON);

    (mzs.iter().map(|&mz| mz + MASS_PROTON).collect(), intensities)
}

pub fn generate_averagine_spectrum(
    mass: f64,
    charge: i32,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    amp: Option<f64>
) -> MzSpectrum {
    let amp = amp.unwrap_or(1e4);
    let lb = mass / charge as f64 - 0.2;
    let ub = mass / charge as f64 + k as f64 + 0.2;

    let (mz, intensities) = generate_isotope_pattern(
        lb,
        ub,
        mass,
        charge as f64,
        amp,
        k as usize,
        0.008492569002123142,
        resolution,
    );

    let spectrum = MzSpectrum::new(mz, intensities).to_resolution(resolution).filter_ranged(lb, ub, min_intensity as f64, 1e9);

    if centroid {
        spectrum.to_centroided(std::cmp::max(min_intensity, 1), 1.0 / 10f64.powi(resolution - 1), true)
    } else {
        spectrum
    }
}

pub fn generate_averagine_spectra(
    masses: Vec<f64>,
    charges: Vec<i32>,
    min_intensity: i32,
    k: i32,
    resolution: i32,
    centroid: bool,
    num_threads: usize,
    amp: Option<f64>
) -> Vec<MzSpectrum> {
    let amp = amp.unwrap_or(1e5);
    let mut spectra: Vec<MzSpectrum> = Vec::new();
    let thread_pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();

    thread_pool.install(|| {
        spectra = masses.par_iter().zip(charges.par_iter()).map(|(&mass, &charge)| {
            generate_averagine_spectrum(mass, charge, min_intensity, k, resolution, centroid, Some(amp))
        }).collect();
    });

    spectra
}

pub fn get_num_protonizable_sites(sequence: &str) -> usize {
    let mut sites = 1; // n-terminus
    for s in sequence.chars() {
        match s {
            'H' | 'R' | 'K' => sites += 1,
            _ => {}
        }
    }
    sites
}

pub fn simulate_charge_state_for_sequence(sequence: &str, max_charge: Option<usize>, charged_probability: Option<f64>) -> Vec<f64> {
    let charged_prob = charged_probability.unwrap_or(0.5);
    let max_charge = max_charge.unwrap_or(5);
    let num_protonizable_sites = get_num_protonizable_sites(sequence);
    let mut charge_state_probs = vec![0.0; max_charge];

    for charge in 0..max_charge {
        let binom = Binomial::new(charged_prob, num_protonizable_sites as u64).unwrap();
        let prob = binom.pmf(charge as u64);

        charge_state_probs[charge] = prob;
    }

    charge_state_probs
}

pub fn simulate_charge_states_for_sequences(sequences: Vec<&str>,  num_threads: usize, max_charge: Option<usize>, charged_probability: Option<f64>) -> Vec<Vec<f64>> {
    let pool = ThreadPoolBuilder::new().num_threads(num_threads).build().unwrap();
    pool.install(|| {
        sequences.par_iter()
            .map(|sequence| simulate_charge_state_for_sequence(sequence, max_charge, charged_probability))
            .collect()
    })
}