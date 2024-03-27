use std::collections::HashMap;

pub struct SumFormula {
    pub formula: String,
    pub elements: HashMap<String, i32>,
}

impl SumFormula {
    pub fn new(formula: &str) -> Self {
        let elements = parse_formula(formula);
        SumFormula {
            formula: formula.to_string(),
            elements,
        }
    }
}

fn is_uppercase_or_digit(s: &str) -> bool {
    s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn parse_formula(formula: &str) -> HashMap<String, i32> {

    assert!(is_uppercase_or_digit(formula), "Invalid formula");

    let mut map = HashMap::new();
    let mut current_element = String::new();
    let mut current_count = String::new();

    let mut chars = formula.chars().peekable();

    while let Some(c) = chars.next() {
        if c.is_ascii_uppercase() {
            // Save previous element and its count
            if !current_element.is_empty() {
                let count = current_count.parse::<i32>().unwrap_or(1); // Default to 1 if no count is provided
                *map.entry(current_element.clone()).or_insert(0) += count;
            }

            // Reset for the new element
            current_element = c.to_string();
            current_count = String::new();
        } else if c.is_ascii_digit() {
            current_count.push(c);
        } else if c.is_ascii_lowercase() {
            current_element.push(c);
        }

        // Look ahead to see if the next character is a new element (uppercase letter)
        // or the end of the string, to handle the last element/count pair
        if chars.peek().map_or(true, |next_c| next_c.is_ascii_uppercase()) {
            let count = current_count.parse::<i32>().unwrap_or(1); // Default to 1 if no count is provided
            *map.entry(current_element.clone()).or_insert(0) += count;

            // Reset for potential new elements
            current_element = String::new();
            current_count = String::new();
        }
    }

    map
}
