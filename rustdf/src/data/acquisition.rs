use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum AcquisitionMode {
    PRECURSOR,
    DDA,
    DIA,
    Unknown,
}

impl AcquisitionMode {
    pub fn to_i32(&self) -> i32 {
        match self {
            AcquisitionMode::PRECURSOR => 0,
            AcquisitionMode::DDA => 8,
            AcquisitionMode::DIA => 9,
            AcquisitionMode::Unknown => -1,
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            AcquisitionMode::PRECURSOR => "PRECURSOR",
            AcquisitionMode::DDA => "DDA",
            AcquisitionMode::DIA => "DIA",
            AcquisitionMode::Unknown => "UNKNOWN",
        }
    }
}

impl Display for AcquisitionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquisitionMode::PRECURSOR => write!(f, "PRECURSOR"),
            AcquisitionMode::DDA => write!(f, "DDA"),
            AcquisitionMode::DIA => write!(f, "DIA"),
            AcquisitionMode::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

impl From<i32> for AcquisitionMode {
    fn from(item: i32) -> Self {
        match item {
            0 => AcquisitionMode::PRECURSOR,
            8 => AcquisitionMode::DDA,
            9 => AcquisitionMode::DIA,
            _ => AcquisitionMode::Unknown,
        }
    }
}

impl From<&str> for AcquisitionMode {
    fn from(item: &str) -> Self {
        match item {
            "PRECURSOR" => AcquisitionMode::PRECURSOR,
            "DDA" => AcquisitionMode::DDA,
            "DIA" => AcquisitionMode::DIA,
            _ => AcquisitionMode::Unknown,
        }
    }
}
