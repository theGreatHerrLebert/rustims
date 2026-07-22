//! Sum-formula parsing — `"C2H3NO"` → element counts → monoisotopic mass.
//!
//! Ported from mscore's parser (positive counts; an uppercase letter starts an element, optional
//! lowercase letters extend it, optional digits give the count). Element masses come from
//! [`crate::elements`], so a formula's mass is built on the same table as everything else — which is
//! exactly what lets [`crate::modification`] cross-check a modification's declared mass against its
//! composition.

use crate::elements;

/// Why a sum formula could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormulaError {
    /// A symbol that is not a known element.
    UnknownElement(String),
    /// A character where an element symbol was expected.
    Unexpected(char),
    /// The formula was empty.
    Empty,
}

impl std::fmt::Display for FormulaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormulaError::UnknownElement(s) => write!(f, "unknown element {s:?}"),
            FormulaError::Unexpected(c) => write!(f, "unexpected character {c:?} in formula"),
            FormulaError::Empty => write!(f, "empty formula"),
        }
    }
}

impl std::error::Error for FormulaError {}

/// Parse a sum formula into `(element, count)` pairs (positive counts only).
pub fn parse(formula: &str) -> Result<Vec<(String, u32)>, FormulaError> {
    let bytes = formula.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if !c.is_ascii_uppercase() {
            return Err(FormulaError::Unexpected(c as char));
        }
        let mut el = String::new();
        el.push(c as char);
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_lowercase() {
            el.push(bytes[i] as char);
            i += 1;
        }
        let mut num = String::new();
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            num.push(bytes[i] as char);
            i += 1;
        }
        let count: u32 = if num.is_empty() { 1 } else { num.parse().unwrap() };
        if elements::monoisotopic_mass(&el).is_none() {
            return Err(FormulaError::UnknownElement(el));
        }
        out.push((el, count));
    }
    if out.is_empty() {
        return Err(FormulaError::Empty);
    }
    Ok(out)
}

/// Monoisotopic mass of a sum formula (Da).
pub fn monoisotopic_mass(formula: &str) -> Result<f64, FormulaError> {
    let mut m = 0.0;
    for (el, n) in parse(formula)? {
        // parse() already checked the element is known
        m += elements::monoisotopic_mass(&el).unwrap() * n as f64;
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_weighs() {
        // phospho HO3P = 79.96633
        assert!((monoisotopic_mass("HO3P").unwrap() - 79.96633).abs() < 1e-4);
        // GG C4H6N2O2 = 114.04293
        assert!((monoisotopic_mass("C4H6N2O2").unwrap() - 114.04293).abs() < 1e-4);
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(monoisotopic_mass(""), Err(FormulaError::Empty));
        assert_eq!(monoisotopic_mass("Xy2"), Err(FormulaError::UnknownElement("Xy".into())));
        assert_eq!(monoisotopic_mass("2C"), Err(FormulaError::Unexpected('2')));
    }
}
