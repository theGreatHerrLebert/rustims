use regex::Regex;
use crate::chemistry::unimod::unimod_modifications_mass;

/// Convert a peptide sequence with UNIMOD annotations to a list of tokens
///
/// # Arguments
///
/// * `sequence` - a string slice of the peptide sequence
/// * `group_modifications` - a boolean indicating whether to group the amino acid before the UNIMOD with the UNIMOD
///
/// # Returns
///
/// * `Vec<String>` - a vector of strings representing the tokens
///
/// # Example
///
/// ```
/// use mscore::chemistry::utility::unimod_sequence_to_tokens;
///
/// let sequence = "PEPTIDE[UNIMOD:1]H";
/// let tokens = unimod_sequence_to_tokens(sequence, false);
/// assert_eq!(tokens, vec!["P", "E", "P", "T", "I", "D", "E", "[UNIMOD:1]", "H"]);
/// let tokens = unimod_sequence_to_tokens(sequence, true);
/// assert_eq!(tokens, vec!["P", "E", "P", "T", "I", "D", "E[UNIMOD:1]", "H"]);
/// ```
pub fn unimod_sequence_to_tokens(sequence: &str, group_modifications: bool) -> Vec<String> {
    let pattern = Regex::new(r"\[UNIMOD:\d+\]").unwrap();
    let mut tokens = Vec::new();
    let mut last_index = 0;

    for mat in pattern.find_iter(sequence) {
        if group_modifications {
            // When grouping, include the amino acid before the UNIMOD in the token
            let pre_mod_sequence = &sequence[last_index..mat.start()];
            let aa_sequence = if pre_mod_sequence.is_empty() {
                ""
            } else {
                &pre_mod_sequence[..pre_mod_sequence.len() - 1]
            };
            tokens.extend(aa_sequence.chars().map(|c| c.to_string()));

            // Group the last amino acid with the UNIMOD as one token
            let grouped_mod = format!("{}{}", pre_mod_sequence.chars().last().unwrap_or_default().to_string(), &sequence[mat.start()..mat.end()]);
            tokens.push(grouped_mod);
        } else {
            // Extract the amino acids before the current UNIMOD and add them as individual tokens
            let aa_sequence = &sequence[last_index..mat.start()];
            tokens.extend(aa_sequence.chars().map(|c| c.to_string()));

            // Add the UNIMOD as its own token
            let unimod = &sequence[mat.start()..mat.end()];
            tokens.push(unimod.to_string());
        }

        // Update last_index to the end of the current UNIMOD
        last_index = mat.end();
    }

    if !group_modifications || last_index < sequence.len() {
        // Add the remaining amino acids after the last UNIMOD as individual tokens
        let remaining_aa_sequence = &sequence[last_index..];
        tokens.extend(remaining_aa_sequence.chars().map(|c| c.to_string()));
    }

    tokens
}

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

pub fn reshape_prosit_array(flat_array: Vec<f64>) -> Vec<Vec<Vec<f64>>> {
    let mut array_return: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 3]; 2]; 29];
    let mut ptr = 0;

    for c in 0..3 {
        for row in 0..29 {
            // Fill in the Y ion values
            array_return[row][0][c] = flat_array[ptr];
            ptr += 1;
        }
        for row in 0..29 {
            // Fill in the B ion values
            array_return[row][1][c] = flat_array[ptr];
            ptr += 1;
        }
    }

    array_return
}