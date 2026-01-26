use pyo3::prelude::*;
use regex::Regex;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::env;

use serde::{Deserialize, Serialize};

/// Standard amino acids (20)
const STANDARD_AA: &str = "ACDEFGHIKLMNPQRSTVWY";

/// Extended amino acids:
/// - U: Selenocysteine
/// - O: Pyrrolysine
/// - B: Asparagine or Aspartic acid (ambiguous)
/// - Z: Glutamine or Glutamic acid (ambiguous)
/// - X: Unknown amino acid
/// - J: Leucine or Isoleucine (ambiguous)
const EXTENDED_AA: &str = "UOBZXJ";

/// All supported amino acids (standard + extended)
const ALL_AA: &str = "ACDEFGHIKLMNPQRSTVWYUOBZXJ";

/// Default common PTM composites - these are tokenized as single units
/// Covers the most frequent modifications in proteomics
const DEFAULT_COMMON_COMPOSITES: &[&str] = &[
    // Cysteine alkylation (most common)
    "C[UNIMOD:4]",      // Carbamidomethyl (iodoacetamide)
    "C[UNIMOD:26]",     // Pyridylethyl
    "C[UNIMOD:39]",     // Methylthio
    "C[UNIMOD:312]",    // Carbamidomethyl-D(3)
    // Methionine oxidation
    "M[UNIMOD:35]",     // Oxidation
    "M[UNIMOD:425]",    // Dioxidation
    // Phosphorylation
    "S[UNIMOD:21]",     // Phospho
    "T[UNIMOD:21]",     // Phospho
    "Y[UNIMOD:21]",     // Phospho
    // Acetylation
    "K[UNIMOD:1]",      // Acetyl
    // Deamidation
    "N[UNIMOD:7]",      // Deamidated
    "Q[UNIMOD:7]",      // Deamidated
    // Ubiquitination
    "K[UNIMOD:121]",    // GlyGly (ubiquitin remnant)
    // Methylation
    "K[UNIMOD:34]",     // Methyl
    "R[UNIMOD:34]",     // Methyl
    "K[UNIMOD:36]",     // Dimethyl
    "R[UNIMOD:36]",     // Dimethyl
    "K[UNIMOD:37]",     // Trimethyl
    // Carbamylation
    "K[UNIMOD:5]",      // Carbamyl
    // Formylation
    "K[UNIMOD:122]",    // Formyl
    // Hydroxylation
    "P[UNIMOD:35]",     // Oxidation (Hydroxyproline)
    "K[UNIMOD:35]",     // Oxidation (Hydroxylysine)
    // SILAC labels
    "K[UNIMOD:259]",    // Label:13C(6)15N(2)
    "R[UNIMOD:267]",    // Label:13C(6)15N(4)
    "K[UNIMOD:188]",    // Label:13C(6)
    "R[UNIMOD:188]",    // Label:13C(6)
    // TMT/iTRAQ labels
    "K[UNIMOD:737]",    // TMT6plex
    "K[UNIMOD:2016]",   // TMTpro
    // Nitrosylation
    "C[UNIMOD:275]",    // Nitrosyl
    // Succinylation
    "K[UNIMOD:64]",     // Succinyl
    // Crotonylation
    "K[UNIMOD:1363]",   // Crotonyl
];

/// Default N-terminal modifications
const DEFAULT_NTERM_MODS: &[&str] = &[
    "[UNIMOD:1]-",      // Acetyl (N-term)
    "[UNIMOD:5]-",      // Carbamyl (N-term)
    "[UNIMOD:28]-",     // Gln->pyro-Glu (N-term Q)
    "[UNIMOD:27]-",     // Glu->pyro-Glu (N-term E)
    "[UNIMOD:385]-",    // Ammonia-loss (N-term C)
    "[UNIMOD:7]-",      // Deamidated (N-term)
    "[UNIMOD:737]-",    // TMT6plex (N-term)
    "[UNIMOD:2016]-",   // TMTpro (N-term)
    "[UNIMOD:36]-",     // Dimethyl (N-term)
    "[UNIMOD:34]-",     // Methyl (N-term)
    "[UNIMOD:122]-",    // Formyl (N-term)
];

/// Default C-terminal modifications
const DEFAULT_CTERM_MODS: &[&str] = &[
    "-[UNIMOD:2]",      // Amidated (C-term)
    "-[UNIMOD:258]",    // Cation:Na (C-term)
];

#[derive(Serialize, Deserialize)]
struct TokenizerState {
    common_composites: Vec<String>,
    terminal_mods: Vec<String>,
    special_tokens: Vec<String>,
    unimod_tokens: Vec<String>,
    vocab: Vec<String>,
}

#[pyclass]
pub struct PyProformaTokenizer {
    common_composites: HashSet<String>,
    terminal_mods: HashSet<String>,
    special_tokens: HashSet<String>,
    unimod_tokens: HashSet<String>,
    aa_tokens: HashSet<String>,
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    unk_token: String,
    pad_token: String,
}

#[pymethods]
impl PyProformaTokenizer {
    /// Create a new tokenizer with the specified vocabulary components.
    ///
    /// # Arguments
    /// * `common_composites` - AA+modification combinations to tokenize as single units
    /// * `terminal_mods` - N-terminal and C-terminal modifications
    /// * `special_tokens` - Special tokens like [PAD], [CLS], [SEP], [MASK], [UNK]
    /// * `unimod_tokens` - Individual UNIMOD tokens for fallback (e.g., "[UNIMOD:123]")
    #[new]
    #[pyo3(signature = (common_composites=None, terminal_mods=None, special_tokens=None, unimod_tokens=None))]
    fn new(
        common_composites: Option<Vec<String>>,
        terminal_mods: Option<Vec<String>>,
        special_tokens: Option<Vec<String>>,
        unimod_tokens: Option<Vec<String>>,
    ) -> Self {
        // Use default special tokens if not provided
        let special_tokens = special_tokens.unwrap_or_else(|| {
            vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ]
        });

        // Use default common composites if not provided
        let common_composites = common_composites.unwrap_or_else(|| {
            DEFAULT_COMMON_COMPOSITES
                .iter()
                .map(|s| s.to_string())
                .collect()
        });

        // Use default terminal mods if not provided (combine N-term and C-term)
        let terminal_mods = terminal_mods.unwrap_or_else(|| {
            DEFAULT_NTERM_MODS
                .iter()
                .chain(DEFAULT_CTERM_MODS.iter())
                .map(|s| s.to_string())
                .collect()
        });

        // Use default UNIMOD tokens if not provided
        // Include all PSI standard UNIMOD IDs (1-2100) for full compatibility
        // This covers the entire PSI-MOD/UNIMOD database
        let unimod_tokens = unimod_tokens.unwrap_or_else(|| {
            (1..=2100)
                .map(|i| format!("[UNIMOD:{}]", i))
                .collect()
        });

        // All amino acids (standard + extended)
        let aa_tokens = ALL_AA
            .chars()
            .map(|c| c.to_string())
            .collect::<HashSet<_>>();

        // Build complete vocabulary
        let mut all_tokens = special_tokens
            .iter()
            .chain(terminal_mods.iter())
            .chain(common_composites.iter())
            .chain(unimod_tokens.iter())
            .chain(aa_tokens.iter())
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        // Remove duplicates and sort for consistent ID assignment
        all_tokens.sort();
        all_tokens.dedup();

        let token_to_id = all_tokens
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), i))
            .collect::<HashMap<_, _>>();

        PyProformaTokenizer {
            common_composites: common_composites.into_iter().collect(),
            terminal_mods: terminal_mods.into_iter().collect(),
            special_tokens: special_tokens.into_iter().collect(),
            unimod_tokens: unimod_tokens.into_iter().collect(),
            aa_tokens,
            token_to_id,
            id_to_token: all_tokens,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
        }
    }

    /// Create a tokenizer with default vocabulary suitable for transformer models.
    ///
    /// This uses sensible defaults for proteomics:
    /// - Standard + extended amino acids (26 total: A-Y + U, O, B, Z, X, J)
    /// - Common PTM composites (~30 common modifications as single tokens)
    /// - N-term and C-term modifications
    /// - All PSI standard UNIMOD tokens (1-2100) for full compatibility
    ///
    /// The resulting vocabulary size is approximately 2200 tokens.
    #[staticmethod]
    fn with_defaults() -> Self {
        Self::new(None, None, None, None)
    }

    /// Tokenize a single peptide sequence
    fn tokenize(&self, seq: &str) -> Vec<String> {
        tokenize_internal(seq, &self.common_composites, &self.terminal_mods)
    }

    /// Tokenize multiple peptide sequences in parallel
    fn tokenize_many(&self, seqs: Vec<String>) -> Vec<Vec<String>> {
        seqs.par_iter()
            .map(|s| tokenize_internal(s, &self.common_composites, &self.terminal_mods))
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
            unimod_tokens: self.unimod_tokens.iter().cloned().collect(),
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

        // All amino acids (standard + extended)
        let aa_tokens = ALL_AA
            .chars()
            .map(|c| c.to_string())
            .collect::<HashSet<_>>();

        let token_to_id = state
            .vocab
            .iter()
            .enumerate()
            .map(|(i, tok)| (tok.clone(), i))
            .collect::<HashMap<_, _>>();

        Ok(PyProformaTokenizer {
            common_composites: state.common_composites.into_iter().collect(),
            terminal_mods: state.terminal_mods.into_iter().collect(),
            special_tokens: state.special_tokens.into_iter().collect(),
            unimod_tokens: state.unimod_tokens.into_iter().collect(),
            aa_tokens,
            token_to_id,
            id_to_token: state.vocab,
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
        })
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get the padding token ID
    fn pad_token_id(&self) -> usize {
        self.token_to_id[&self.pad_token]
    }

    /// Get the unknown token ID
    fn unk_token_id(&self) -> usize {
        self.token_to_id[&self.unk_token]
    }

    /// Get the CLS token ID
    fn cls_token_id(&self) -> PyResult<usize> {
        self.token_to_id
            .get("[CLS]")
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("[CLS] token not in vocabulary"))
    }

    /// Get the SEP token ID
    fn sep_token_id(&self) -> PyResult<usize> {
        self.token_to_id
            .get("[SEP]")
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("[SEP] token not in vocabulary"))
    }

    /// Check if a token is in the vocabulary
    fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get the ID for a specific token, or None if not found
    fn get_token_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }
}

/// Tokenization logic (pure function)
///
/// Tokenizes a peptide sequence with UNIMOD modifications.
///
/// # Arguments
/// * `seq` - Peptide sequence in UNIMOD format (e.g., "[UNIMOD:1]-MC[UNIMOD:4]PEPTIDE-[UNIMOD:2]")
/// * `common_composites` - Set of AA+modification combinations to tokenize as single units
/// * `terminal_mods` - Set of terminal modifications (for checking composites)
///
/// # Returns
/// Vector of tokens including [CLS] prefix and [SEP] suffix
///
/// # Tokenization Strategy
/// 1. Extract N-terminal modification if present
/// 2. Extract C-terminal modification if present
/// 3. Process amino acids left to right:
///    - If AA+mod is in common_composites: emit as single token
///    - Otherwise: emit AA and mod as separate tokens
/// 4. Wrap with [CLS] and [SEP]
fn tokenize_internal(
    seq: &str,
    common_composites: &HashSet<String>,
    terminal_mods: &HashSet<String>,
) -> Vec<String> {
    let mut tokens: Vec<String> = vec![];

    // Regex patterns for PROFORMA-style UNIMOD notation
    let nterm_re = Regex::new(r"^\[UNIMOD:\d+]-").unwrap();
    let cterm_re = Regex::new(r"-\[UNIMOD:\d+]$").unwrap();
    let aa_re = Regex::new(r"([A-Z])(\[UNIMOD:\d+])?").unwrap();

    let mut seq = seq.to_string();

    // Handle N-terminal modification
    if let Some(nterm) = nterm_re.find(&seq) {
        let nterm_str = nterm.as_str().to_string();
        // Check if it's a known terminal mod composite
        if terminal_mods.contains(&nterm_str) {
            tokens.push(nterm_str);
        } else {
            // Just the UNIMOD part without the trailing dash
            tokens.push(nterm.as_str().trim_end_matches('-').to_string());
        }
        seq = seq[nterm.end()..].to_string();
    }

    // Handle C-terminal modification
    let mut cterm_token = None;
    if let Some(cterm) = cterm_re.find(&seq) {
        let cterm_str = cterm.as_str().to_string();
        // Check if it's a known terminal mod composite
        if terminal_mods.contains(&cterm_str) {
            cterm_token = Some(cterm_str);
        } else {
            // Just the UNIMOD part without the leading dash
            cterm_token = Some(cterm.as_str().trim_start_matches('-').to_string());
        }
        seq = seq[..cterm.start()].to_string();
    }

    // Process amino acids and their modifications
    for cap in aa_re.captures_iter(&seq) {
        let aa = cap.get(1).unwrap().as_str();
        if let Some(mod_group) = cap.get(2) {
            let composite = format!("{}{}", aa, mod_group.as_str());
            if common_composites.contains(&composite) {
                // Known composite: emit as single token
                tokens.push(composite);
            } else {
                // Unknown composite: emit AA and mod separately
                tokens.push(aa.to_string());
                tokens.push(mod_group.as_str().to_string());
            }
        } else {
            tokens.push(aa.to_string());
        }
    }

    // Add C-terminal token at the end (before SEP)
    if let Some(cterm) = cterm_token {
        tokens.push(cterm);
    }

    // Wrap with special tokens
    tokens.insert(0, "[CLS]".to_string());
    tokens.push("[SEP]".to_string());

    tokens
}

#[pymodule]
pub fn py_ml_utility(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProformaTokenizer>()?;
    Ok(())
}