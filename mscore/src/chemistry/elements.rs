use std::collections::HashMap;

/// Atomic Weights
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, f64>` - a map of atomic symbols to their monoisotopic weights
///
/// # Example
///
/// ```
/// use mscore::chemistry::elements::atomic_weights_mono_isotopic;
///
/// let atomic_weights = atomic_weights_mono_isotopic();
/// assert_eq!(atomic_weights.get("H"), Some(&1.00782503223));
/// ```
pub fn atomic_weights_mono_isotopic() -> HashMap<&'static str, f64> {
    // R1 fold: element masses come from ms-chem (single source of truth). Identical values
    // (ms-chem was ported from this table), so zero output change.
    ms_chem::elements::ELEMENTS.iter().copied().collect()
}

/// Isotopic Weights
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, Vec<f64>>` - a map of atomic symbols to their isotopic weights
///
/// # Example
///
/// ```
/// use mscore::chemistry::elements::atoms_isotopic_weights;
///
/// let isotopic_weights = atoms_isotopic_weights();
/// assert_eq!(isotopic_weights.get("H"), Some(&vec![1.00782503223, 2.01410177812]));
/// ```
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

/// Isotopic Abundance
///
/// # Arguments
///
/// None
///
/// # Returns
///
/// * `HashMap<&'static str, Vec<f64>>` - a map of atomic symbols to their isotopic abundances
///
/// # Example
///
/// ```
/// use mscore::chemistry::elements::isotopic_abundance;
///
/// let isotopic_abundance = isotopic_abundance();
/// assert_eq!(isotopic_abundance.get("H"), Some(&vec![0.999885, 0.000115]));
/// ```
pub fn isotopic_abundance() -> HashMap<&'static str, Vec<f64>> {

    let mut map = HashMap::new();

    map.insert("H", vec![0.999885, 0.000115]);
    map.insert("He", vec![0.99999866, 0.00000134]);
    map.insert("Li", vec![0.0759, 0.9241]);
    map.insert("Be", vec![1.0]);
    map.insert("B", vec![0.199, 0.801]);
    map.insert("C", vec![0.9893, 0.0107]);
    map.insert("N", vec![0.99636, 0.00364]); // CIAAW (R1 fold step 5)
    map.insert("O", vec![0.99757, 0.00038, 0.00205]);
    map.insert("F", vec![1.0]);
    map.insert("Ne", vec![0.9048, 0.0027, 0.0925]);
    map.insert("Na", vec![0.5429, 0.4571]);
    map.insert("Mg", vec![0.7899, 0.1000, 0.1101]);
    map.insert("Al", vec![1.0]);
    map.insert("Si", vec![0.9223, 0.0467, 0.0310]);
    map.insert("P", vec![1.0]);
    map.insert("S", vec![0.9499, 0.0075, 0.0425]); // CIAAW (R1 fold step 5)
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