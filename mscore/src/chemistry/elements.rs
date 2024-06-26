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