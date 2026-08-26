use std::collections::HashMap;
use crate::algorithm::isotope::generate_isotope_distribution;
use crate::chemistry::constants::MASS_PROTON;
use crate::chemistry::elements::atomic_weights_mono_isotopic;
use crate::data::spectrum::MzSpectrum;

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
    pub fn monoisotopic_weight(&self) -> f64 {
        let atomic_weights = atomic_weights_mono_isotopic();

        // Iterate elements in a deterministic (sorted) order. `elements` is a
        // HashMap whose iteration order is randomized per instance, and
        // floating-point addition is not associative, so an unsorted fold can
        // return masses that differ in their least-significant bits between
        // calls. Sorting pins one canonical accumulation order, matching what
        // `generate_isotope_distribution` already does for the same map.
        let mut elements: Vec<(&String, i32)> =
            self.elements.iter().map(|(k, v)| (k, *v)).collect();
        elements.sort_unstable_by(|a, b| a.0.cmp(b.0));

        elements.into_iter().fold(0.0, |acc, (element, count)| {
            acc + atomic_weights[element.as_str()] * count as f64
        })
    }

    pub fn isotope_distribution(&self, charge: i32) -> MzSpectrum {
        let distribution = generate_isotope_distribution(&self.elements, 1e-3, 1e-9, 200);
        let intensity = distribution.iter().map(|(_, i)| *i).collect();
        let mz = distribution.iter().map(|(m, _)| (*m + charge as f64 * MASS_PROTON) / charge as f64).collect();
        MzSpectrum::new(mz, intensity)
    }
}

fn parse_formula(formula: &str) -> Result<HashMap<String, i32>, String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    // A formula whose element masses are genuinely order-sensitive: summing its
    // five elements in different orders yields two distinct f64 bit patterns
    // (a 66/54 split over the 120 permutations), so a randomized HashMap order
    // reliably produces both within this many draws. Small formulas such as
    // C6H12O6 round identically under every permutation and cannot catch the bug.
    const ORDER_SENSITIVE_FORMULA: &str = "C100H150N20O30S2";

    #[test]
    fn monoisotopic_weight_is_bitwise_stable() {
        let mut observed_bits = BTreeSet::new();
        for _ in 0..1024 {
            let formula = SumFormula::new(ORDER_SENSITIVE_FORMULA);
            observed_bits.insert(formula.monoisotopic_weight().to_bits());
        }

        assert_eq!(
            observed_bits.len(),
            1,
            "identical sum formula produced multiple exact monoisotopic weights: {observed_bits:?}"
        );
    }

    #[test]
    fn monoisotopic_weight_matches_expected_mass() {
        let formula = SumFormula::new("C6H12O6");
        let quantized = (formula.monoisotopic_weight() * 1e6).round() as i64;
        assert_eq!(quantized, 180063388);
    }
}
