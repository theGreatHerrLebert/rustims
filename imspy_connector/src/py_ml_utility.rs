use pyo3::prelude::*;
use regex::Regex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct TokenizerState {
    common_composites: Vec<String>,
    terminal_mods: Vec<String>,
    special_tokens: Vec<String>,
    vocab: Vec<String>,
}

#[pyclass]
pub struct PyTokenizer {
    common_composites: HashSet<String>,
    terminal_mods: HashSet<String>,
    special_tokens: HashSet<String>,
    aa_tokens: HashSet<String>,
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_token: String,
    pad_token: String,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(
        common_composites: Vec<String>,
        terminal_mods: Vec<String>,
        special_tokens: Vec<String>,
    ) -> Self {
        let aa_tokens = "ACDEFGHIKLMNPQRSTVWY"
            .chars()
            .map(|c| c.to_string())
            .collect::<HashSet<_>>();

        let mut all_tokens = special_tokens
            .iter()
            .chain(terminal_mods.iter())
            .chain(common_composites.iter())
            .chain(aa_tokens.iter())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        all_tokens.sort(); // consistent ID assignment

        let token_to_id = all_tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), i))
            .collect::<HashMap<_, _>>();

        PyTokenizer {
            common_composites: common_composites.into_iter().collect(),
            terminal_mods: terminal_mods.into_iter().collect(),
            special_tokens: special_tokens.into_iter().collect(),
            aa_tokens,
            token_to_id,
            id_to_token: all_tokens,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
        }
    }

    fn tokenize(&self, seq: &str) -> Vec<String> {
        tokenize_internal(seq, &self.common_composites)
    }

    fn tokenize_many(&self, seqs: Vec<String>) -> Vec<Vec<String>> {
        seqs.par_iter()
            .map(|s| tokenize_internal(s, &self.common_composites))
            .collect()
    }

    fn encode(&self, tokens: Vec<String>) -> Vec<usize> {
        tokens
            .into_iter()
            .map(|tok| {
                *self
                    .token_to_id
                    .get(&tok)
                    .unwrap_or(&self.token_to_id[&self.unk_token])
            })
            .collect()
    }

    fn decode(&self, ids: Vec<usize>) -> Vec<String> {
        ids.into_iter()
            .map(|i| {
                self.id_to_token
                    .get(i)
                    .cloned()
                    .unwrap_or(self.unk_token.clone())
            })
            .collect()
    }

    fn encode_many(&self, token_batches: Vec<Vec<String>>, pad: bool) -> (Vec<Vec<usize>>, Vec<Vec<u8>>) {
        let encoded: Vec<Vec<usize>> = token_batches
            .into_iter()
            .map(|tokens| self.encode(tokens))
            .collect();

        if !pad {
            let masks = encoded
                .iter()
                .map(|seq| vec![1u8; seq.len()])
                .collect::<Vec<_>>();
            return (encoded, masks);
        }

        let max_len = encoded.iter().map(|seq| seq.len()).max().unwrap_or(0);
        let pad_id = self.token_to_id[&self.pad_token];

        let mut padded = Vec::with_capacity(encoded.len());
        let mut masks = Vec::with_capacity(encoded.len());

        for mut seq in encoded {
            let pad_len = max_len - seq.len();
            let mut mask = vec![1u8; seq.len()];
            seq.extend(std::iter::repeat(pad_id).take(pad_len));
            mask.extend(std::iter::repeat(0u8).take(pad_len));
            padded.push(seq);
            masks.push(mask);
        }

        (padded, masks)
    }

    fn decode_many(&self, id_batches: Vec<Vec<usize>>) -> Vec<Vec<String>> {
        id_batches
            .into_iter()
            .map(|ids| self.decode(ids))
            .collect()
    }

    fn get_vocab(&self) -> Vec<String> {
        self.id_to_token.clone()
    }

    fn set_num_threads(&self, num: usize) {
        env::set_var("RAYON_NUM_THREADS", num.to_string());
    }

    #[pyo3(name = "save_vocab")]
    fn save_vocab_py(&self, path: &str) -> PyResult<()> {
        let state = TokenizerState {
            common_composites: self.common_composites.iter().cloned().collect(),
            terminal_mods: self.terminal_mods.iter().cloned().collect(),
            special_tokens: self.special_tokens.iter().cloned().collect(),
            vocab: self.id_to_token.clone(),
        };

        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }

    #[staticmethod]
    #[pyo3(name = "load_vocab")]
    fn load_vocab_py(path: &str) -> PyResult<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let state: TokenizerState = serde_json::from_str(&json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let aa_tokens = "ACDEFGHIKLMNPQRSTVWY"
            .chars()
            .map(|c| c.to_string())
            .collect::<HashSet<_>>();

        let token_to_id = state
            .vocab
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), i))
            .collect::<HashMap<_, _>>();

        Ok(PyTokenizer {
            common_composites: state.common_composites.into_iter().collect(),
            terminal_mods: state.terminal_mods.into_iter().collect(),
            special_tokens: state.special_tokens.into_iter().collect(),
            aa_tokens,
            token_to_id,
            id_to_token: state.vocab,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
        })
    }
}

/// Tokenization logic (pure function)
fn tokenize_internal(seq: &str, common_composites: &HashSet<String>) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];

    let nterm_re = Regex::new(r"^\[UNIMOD:\d+]-").unwrap();
    let cterm_re = Regex::new(r"-\[UNIMOD:\d+]$").unwrap();
    let aa_re = Regex::new(r"([A-Z])(\[UNIMOD:\d+])?").unwrap();

    let mut seq = seq.to_string();

    if let Some(nterm) = nterm_re.find(&seq) {
        tokens.push(nterm.as_str().trim_end_matches('-').to_string());
        seq = seq[nterm.end()..].to_string();
    }

    let mut cterm_token = None;
    if let Some(cterm) = cterm_re.find(&seq) {
        cterm_token = Some(cterm.as_str().trim_start_matches('-').to_string());
        seq = seq[..cterm.start()].to_string();
    }

    for cap in aa_re.captures_iter(&seq) {
        let aa = cap.get(1).unwrap().as_str();
        if let Some(mod_group) = cap.get(2) {
            let composite = format!("{}{}", aa, mod_group.as_str());
            if common_composites.contains(&composite) {
                tokens.push(composite);
            } else {
                tokens.push(aa.to_string());
                tokens.push(mod_group.as_str().to_string());
            }
        } else {
            tokens.push(aa.to_string());
        }
    }

    if let Some(cterm) = cterm_token {
        tokens.push(cterm);
    }

    tokens.insert(0, "[CLS]".to_string());
    tokens.push("[SEP]".to_string());

    tokens
}

#[pymodule]
pub fn py_ml_utility(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    Ok(())
}