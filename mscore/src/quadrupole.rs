use std::f64::consts::E;
use crate::{ImsFrame, MzSpectrum, TimsFrame};

// step function for quadrupole selection simulation
fn smooth_step(x: &Vec<f64>, up_start: f64, up_end: f64, k: f64) -> Vec<f64> {
    let m = (up_start + up_end) / 2.0;
    x.iter().map(|&xi| 1.0 / (1.0 + E.powf(-k * (xi - m)))).collect()
}

// step function for quadrupole selection simulation, going up and down
fn smooth_step_up_down(x: &Vec<f64>, up_start: f64, up_end: f64, down_start: f64, down_end: f64, k: f64) -> Vec<f64> {
    let step_up = smooth_step(x, up_start, up_end, k);
    let step_down = smooth_step(x, down_start, down_end, k);
    step_up.iter().zip(step_down.iter()).map(|(&u, &d)| u - d).collect()
}

// parameterize a step function for quadrupole selection simulation and return a function
pub fn ion_transition_function_midpoint(midpoint: f64, window_length: f64, k: f64) -> impl Fn(Vec<f64>) -> Vec<f64> {
    let half_window = window_length / 2.0;
    let quarter_window = window_length / 4.0;

    let up_start = midpoint - half_window;
    let up_end = midpoint - quarter_window;
    let down_start = midpoint + quarter_window;
    let down_end = midpoint + half_window;

    // take a vector of mz values to their transmission probability
    move |mz: Vec<f64>| -> Vec<f64> {
        smooth_step_up_down(&mz, up_start, up_end, down_start, down_end, k)
    }
}

pub trait IonTransmission {
    fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>) -> Vec<f64>;

    fn filter_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: MzSpectrum, min_probability: Option<f64>) -> MzSpectrum {

        let probability_cutoff = min_probability.unwrap_or(0.5);

        let mz = spectrum.mz.clone();
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &mz);

        let mut filtered_mz = Vec::new();
        let mut filtered_intensity = Vec::new();

        // zip mz and intensity with transmission probability and filter out all mz values with transmission probability 0.001
        for (i, (mz, intensity)) in spectrum.mz.iter().zip(spectrum.intensity.iter()).enumerate() {
            if transmission_probability[i] > probability_cutoff {
                filtered_mz.push(*mz);
                filtered_intensity.push(*intensity);
            }
        }

        MzSpectrum {
            mz: filtered_mz,
            intensity: filtered_intensity,
        }
    }

    fn filter_tims_frame(&self, frame: &TimsFrame) -> TimsFrame {
        let spectra = frame.to_tims_spectra();
        let mut filtered_spectra = Vec::new();

        for mut spectrum in spectra {
            let filtered_spectrum = self.filter_spectrum(frame.frame_id, spectrum.scan, spectrum.spectrum.mz_spectrum);
            if filtered_spectrum.mz.len() > 0 {
                spectrum.spectrum.mz_spectrum = filtered_spectrum;
                filtered_spectra.push(spectrum);
            }
        }

        if  filtered_spectra.len() > 0 {
            TimsFrame::from_tims_spectra(filtered_spectra)
        } else {
            TimsFrame::new(
                frame.frame_id,
                frame.ms_type.clone(),
                0.0,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![]
            )
        }
    }
}