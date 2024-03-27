use std::collections::HashMap;
use crate::chemistry::elements::atomic_weights_mono_isotopic;

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