use regex::Regex;
use crate::chemistry::unimod::unimod_modifications_mass;

pub fn find_unimod_patterns(input_string: &str) -> (String, Vec<f64>) {
    let results = extract_unimod_patterns(input_string);
    let stripped_sequence = remove_unimod_annotation(input_string);
    let index_list = generate_index_list(&results, input_string);
    let mods = calculate_modifications(&index_list, &stripped_sequence);
    (stripped_sequence, mods)
}

fn remove_unimod_annotation(sequence: &str) -> String {
    let pattern = Regex::new(r"\[UNIMOD:\d+]").unwrap();
    pattern.replace_all(sequence, "").to_string()
}

fn extract_unimod_patterns(input_string: &str) -> Vec<(usize, usize, String)> {
    let pattern = Regex::new(r"\[UNIMOD:\d+]").unwrap();
    pattern.find_iter(input_string)
        .map(|mat| (mat.start(), mat.end(), mat.as_str().to_string()))
        .collect()
}

fn generate_index_list(results: &[(usize, usize, String)], sequence: &str) -> Vec<(usize, String)> {
    let mut index_list = Vec::new();
    let mut chars_removed_counter = 0;

    for (start, end, _) in results {
        let num_chars_removed = end - start;
        let mod_str = &sequence[*start..*end];

        let later_aa_index = if *start != 0 {
            start - 1 - chars_removed_counter
        } else {
            0
        };

        index_list.push((later_aa_index, mod_str.to_string()));
        chars_removed_counter += num_chars_removed;
    }

    index_list
}

fn calculate_modifications(index_list: &[(usize, String)], stripped_sequence: &str) -> Vec<f64> {
    let mut mods = vec![0.0; stripped_sequence.len()];
    for (index, mod_str) in index_list {
        if let Some(mass) = unimod_modifications_mass().get(mod_str.as_str()) {
            mods[*index] += mass;
        }
    }
    mods
}