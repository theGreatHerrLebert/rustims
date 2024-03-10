use crate::chemistry::constants::MASS_PROTON;

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
/// use mscore::chemistry::formulas::one_over_reduced_mobility_to_ccs;
///
/// let ccs = one_over_reduced_mobility_to_ccs(0.5, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(ccs, 201.64796734428452);
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
    let reduced_mobility = 1.0 / one_over_k0;
    let reduced_mass = (mz * charge as f64 * mass_gas) / (mz * charge as f64 + mass_gas);
    summary_constant * charge as f64 / (reduced_mass * (temp + t_diff)).sqrt() / reduced_mobility
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
/// use mscore::chemistry::formulas::ccs_to_one_over_reduced_mobility;
///
/// let k0 = ccs_to_one_over_reduced_mobility(806.5918693771381, 1000.0, 2, 28.013, 31.85, 273.15);
/// assert_eq!(k0, 2.0);
/// ```
pub fn ccs_to_one_over_reduced_mobility(
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
/// use mscore::chemistry::formulas::calculate_mz;
///
/// let mz = calculate_mz(1000.0, 2);
/// assert_eq!(mz, 501.007276466621);
/// ```
pub fn calculate_mz(monoisotopic_mass: f64, charge: i32) -> f64 {
    (monoisotopic_mass + charge as f64 * MASS_PROTON) / charge as f64
}