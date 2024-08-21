use std::collections::HashMap;
use crate::algorithm::isotope::generate_isotope_distribution;
use crate::chemistry::constants::MASS_PROTON;
use crate::chemistry::element::atomic_weights_mono_isotopic;
use crate::ms::spectrum::MzSpectrum;

pub struct SumFormula {
    pub formula: String,
    pub elements: HashMap<String, i32>,
}

impl SumFormula {
    pub fn new(formula: &str) -> Self {
        let elements = parse_formula(formula).unwrap();
        SumFormula {
            formula: formula.to_string(),
            elements,
        }
    }
    /// Calculate the monoisotopic weight of the chemical formula.
    ///
    /// Arguments:
    ///
    /// None
    ///
    /// Returns:
    ///
    /// * `f64` - The monoisotopic weight of the chemical formula.
    ///
    /// # Example
    ///
    /// ```
    /// use rustms::chemistry::sum_formula::SumFormula;
    ///
    /// let formula = "H2O";
    /// let sum_formula = SumFormula::new(formula);
    /// assert_eq!(sum_formula.monoisotopic_weight(), 18.01056468403);
    /// ```
    pub fn monoisotopic_weight(&self) -> f64 {
        let atomic_weights = atomic_weights_mono_isotopic();
        self.elements.iter().fold(0.0, |acc, (element, count)| {
            acc + atomic_weights[element.as_str()] * *count as f64
        })
    }

    /// Generate the isotope distribution of the chemical formula.
    ///
    /// Arguments:
    ///
    /// * `charge` - The charge state of the ion.
    ///
    /// Returns:
    ///
    /// * `MzSpectrum` - The isotope distribution of the chemical formula.
    ///
    /// # Example
    ///
    /// ```
    /// use rustms::chemistry::sum_formula::SumFormula;
    /// use rustms::ms::spectrum::MzSpectrum;
    ///
    /// let formula = "C6H12O6";
    /// let sum_formula = SumFormula::new(formula);
    /// let isotope_distribution = sum_formula.isotope_distribution(1);
    /// let mut first_mz = *isotope_distribution.mz.first().unwrap();
    /// // round to first 5 decimal places
    /// first_mz = (first_mz * 1e5).round() / 1e5;
    /// assert_eq!(first_mz, 181.07066);
    /// ```
    pub fn isotope_distribution(&self, charge: i32) -> MzSpectrum {
        let distribution = generate_isotope_distribution(&self.elements, 1e-3, 1e-9, 200);
        let intensity = distribution.iter().map(|(_, i)| *i).collect();
        let mz = distribution.iter().map(|(m, _)| (*m + charge as f64 * MASS_PROTON) / charge as f64).collect();
        MzSpectrum::new(mz, intensity)
    }
}

/// Parse a chemical formula into a map of elements and their counts.
///
/// Arguments:
///
/// * `formula` - The chemical formula to parse.
///
/// Returns:
///
/// * `Result<HashMap<String, i32>, String>` - A map of elements and their counts.
///
/// # Example
///
/// ```
/// use rustms::chemistry::sum_formula::parse_formula;
///
/// let formula = "H2O";
/// let elements = parse_formula(formula).unwrap();
/// assert_eq!(elements.get("H"), Some(&2));
/// assert_eq!(elements.get("O"), Some(&1));
/// ```
pub fn parse_formula(formula: &str) -> Result<HashMap<String, i32>, String> {
    let atomic_weights = atomic_weights_mono_isotopic();
    let mut element_counts = HashMap::new();
    let mut current_element = String::new();
    let mut current_count = String::new();
    let mut chars = formula.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_uppercase() {
            if !current_element.is_empty() {
                let count = current_count.parse::<i32>().unwrap_or(1);
                if atomic_weights.contains_key(current_element.as_str()) {
                    *element_counts.entry(current_element.clone()).or_insert(0) += count;
                } else {
                    return Err(format!("Unknown element: {}", current_element));
                }
            }
            current_element = c.to_string();
            current_count = String::new();
        } else if c.is_ascii_digit() {
            current_count.push(c);
        } else if c.is_ascii_lowercase() {
            current_element.push(c);
        }

        if chars.peek().map_or(true, |next_c| next_c.is_ascii_uppercase()) {
            let count = current_count.parse::<i32>().unwrap_or(1);
            if atomic_weights.contains_key(current_element.as_str()) {
                *element_counts.entry(current_element.clone()).or_insert(0) += count;
            } else {
                return Err(format!("Unknown element: {}", current_element));
            }
            current_element = String::new();
            current_count = String::new();
        }
    }

    Ok(element_counts)
}