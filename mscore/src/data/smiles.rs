#[derive(Debug)]
pub struct Modification {
    pub unimod_id: usize, // e.g., 21 for UNIMOD:21
}

#[derive(Debug)]
pub struct AminoAcidResidue {
    pub aa: char,                      // 'A', 'C', 'D', etc.
    pub modification: Option<Modification>, // None or Some(Modification)
}

#[derive(Debug)]
pub struct PeptideRepresentation {
    pub n_term_mod: Option<Modification>, // Optional N-terminal mod
    pub n_term_residue: AminoAcidResidue,           // First amino acid
    pub core_residues: Vec<AminoAcidResidue>,       // Middle residues
    pub c_term_residue: AminoAcidResidue,           // Last amino acid
    pub c_term_mod: Option<Modification>,  // Optional C-terminal mod
}

impl PeptideRepresentation {
    pub fn new(
        n_term_mod: Option<Modification>,
        n_term_residue: AminoAcidResidue,
        core_residues: Vec<AminoAcidResidue>,
        c_term_residue: AminoAcidResidue,
        c_term_mod: Option<Modification>,
    ) -> Self {
        Self {
            n_term_mod,
            n_term_residue,
            core_residues,
            c_term_residue,
            c_term_mod,
        }
    }
    
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        if let Some(modification) = &self.n_term_mod {
            result.push_str(&format!("[UNIMOD:{}]-", modification.unimod_id));
        }

        result.push(self.n_term_residue.aa);

        for residue in &self.core_residues {
            if let Some(modification) = &residue.modification {
                result.push_str(&format!("[UNIMOD:{}]", modification.unimod_id));
            }
            result.push(residue.aa);
        }

        result.push(self.c_term_residue.aa);

        if let Some(modification) = &self.c_term_mod {
            result.push_str(&format!("-[UNIMOD:{}]", modification.unimod_id));
        }

        result
    }
    
    pub fn from_string(input: &str) -> Result<Self, String> {
        parse_peptide(input)
    }
}


pub fn parse_peptide(input: &str) -> Result<PeptideRepresentation, String> {
    let mut remaining = input.trim();

    let mut n_term_mod = None;
    let mut c_term_mod = None;

    // Handle N-terminal modification
    if remaining.starts_with('[') {
        if let Some(end) = remaining.find("]-") {
            let mod_str = &remaining[1..end]; // inside [ ]
            if !mod_str.is_empty() {
                n_term_mod = Some(parse_modification(mod_str)?);
            }
            remaining = &remaining[end+2..]; // Skip "]‑"
        } else {
            return Err("Invalid N-terminal modification syntax".to_string());
        }
    }

    // Handle C-terminal modification
    if let Some(pos) = remaining.rfind("-[") {
        if remaining.ends_with(']') {
            let mod_str = &remaining[pos+2..remaining.len()-1]; // inside [ ]
            if !mod_str.is_empty() {
                c_term_mod = Some(parse_modification(mod_str)?);
            }
            remaining = &remaining[..pos]; // cut away "-[mod]"
        } else {
            return Err("Invalid C-terminal modification syntax".to_string());
        }
    }

    // Now parse sequence and per-residue mods
    let mut chars = remaining.chars().peekable();
    let mut residues = Vec::new();

    while let Some(c) = chars.next() {
        if c == '[' {
            return Err("Unexpected '[' before amino acid".to_string());
        }

        if !c.is_ascii_alphabetic() {
            return Err(format!("Unexpected character '{}'", c));
        }

        let mut modification = None;

        if let Some('[') = chars.peek() {
            chars.next(); // consume '['

            let mut mod_buf = String::new();
            while let Some(&ch) = chars.peek() {
                if ch == ']' {
                    chars.next(); // consume ']'
                    break;
                }
                mod_buf.push(ch);
                chars.next();
            }

            if !mod_buf.is_empty() {
                modification = Some(parse_modification(&mod_buf)?);
            }
        }

        residues.push(AminoAcidResidue { aa: c, modification });
    }

    if residues.len() < 2 {
        return Err("Peptide must have at least two residues.".to_string());
    }

    let n_term_residue = residues.remove(0);
    let c_term_residue = residues.pop().unwrap();
    let core_residues = residues;

    Ok(PeptideRepresentation {
        n_term_mod,
        n_term_residue,
        core_residues,
        c_term_residue,
        c_term_mod,
    })
}

fn parse_modification(mod_str: &str) -> Result<Modification, String> {
    if let Some(rest) = mod_str.strip_prefix("UNIMOD:") {
        rest.parse::<usize>()
            .map(|id| Modification { unimod_id: id })
            .map_err(|_| format!("Invalid UNIMOD ID: {}", mod_str))
    } else if let Some(rest) = mod_str.strip_prefix("U:") {
        rest.parse::<usize>()
            .map(|id| Modification { unimod_id: id })
            .map_err(|_| format!("Invalid U: ID: {}", mod_str))
    } else {
        Err(format!("Unsupported modification format: {}", mod_str))
    }
}