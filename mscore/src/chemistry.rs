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
