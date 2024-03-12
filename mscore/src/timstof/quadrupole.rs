use std::collections::HashMap;
use std::f64;
use std::f64::consts::E;
use crate::data::spectrum::MzSpectrum;
use crate::timstof::frame::TimsFrame;

/// Sigmoide step function for quadrupole selection simulation
///
/// Arguments:
///
/// * `x` - mz values
/// * `up_start` - start of the step
/// * `up_end` - end of the step
/// * `k` - steepness of the step
///
/// Returns:
///
/// * `Vec<f64>` - transmission probability for each mz value
///
/// # Examples
///
/// ```
/// use mscore::timstof::quadrupole::smooth_step;
///
/// let mz = vec![100.0, 200.0, 300.0];
/// let transmission = smooth_step(&mz, 150.0, 250.0, 0.5).iter().map(
/// |&x| (x * 100.0).round() / 100.0).collect::<Vec<f64>>();
/// assert_eq!(transmission, vec![0.0, 0.5, 1.0]);
/// ```
pub fn smooth_step(x: &Vec<f64>, up_start: f64, up_end: f64, k: f64) -> Vec<f64> {
    let m = (up_start + up_end) / 2.0;
    x.iter().map(|&xi| 1.0 / (1.0 + E.powf(-k * (xi - m)))).collect()
}

/// Sigmoide step function for quadrupole selection simulation
///
/// Arguments:
///
/// * `x` - mz values
/// * `up_start` - start of the step up
/// * `up_end` - end of the step up
/// * `down_start` - start of the step down
/// * `down_end` - end of the step down
/// * `k` - steepness of the step
///
/// Returns:
///
/// * `Vec<f64>` - transmission probability for each mz value
///
/// # Examples
///
/// ```
/// use mscore::timstof::quadrupole::smooth_step_up_down;
///
/// let mz = vec![100.0, 200.0, 300.0];
/// let transmission = smooth_step_up_down(&mz, 150.0, 200.0, 250.0, 300.0, 0.5).iter().map(
/// |&x| (x * 100.0).round() / 100.0).collect::<Vec<f64>>();
/// assert_eq!(transmission, vec![0.0, 1.0, 0.0]);
/// ```
pub fn smooth_step_up_down(x: &Vec<f64>, up_start: f64, up_end: f64, down_start: f64, down_end: f64, k: f64) -> Vec<f64> {
    let step_up = smooth_step(x, up_start, up_end, k);
    let step_down = smooth_step(x, down_start, down_end, k);
    step_up.iter().zip(step_down.iter()).map(|(&u, &d)| u - d).collect()
}

/// Ion transmission function for quadrupole selection simulation
///
/// Arguments:
///
/// * `midpoint` - center of the step
/// * `window_length` - length of the step
/// * `k` - steepness of the step
///
/// Returns:
///
/// * `impl Fn(Vec<f64>) -> Vec<f64>` - ion transmission function
///
/// # Examples
///
/// ```
/// use mscore::timstof::quadrupole::ion_transition_function_midpoint;
///
/// let ion_transmission = ion_transition_function_midpoint(150.0, 50.0, 1.0);
/// let mz = vec![100.0, 150.0, 170.0];
/// let transmission = ion_transmission(mz).iter().map(
/// |&x| (x * 100.0).round() / 100.0).collect::<Vec<f64>>();
/// assert_eq!(transmission, vec![0.0, 1.0, 1.0]);
/// ```
pub fn ion_transition_function_midpoint(midpoint: f64, window_length: f64, k: f64) -> impl Fn(Vec<f64>) -> Vec<f64> {
    let half_window = window_length / 2.0;
    let end_window = window_length / 16.0;

    let up_start = midpoint - half_window;
    let up_end = midpoint - end_window;
    let down_start = midpoint + end_window;
    let down_end = midpoint + half_window;

    // take a vector of mz values to their transmission probability
    move |mz: Vec<f64>| -> Vec<f64> {
        smooth_step_up_down(&mz, up_start, up_end, down_start, down_end, k)
    }
}

pub fn apply_transmission(midpoint: f64, window_length: f64, k: f64, mz: Vec<f64>) -> Vec<f64> {
    ion_transition_function_midpoint(midpoint, window_length, k)(mz)
}

pub trait IonTransmission {
    fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>) -> Vec<f64>;

    fn transmit_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: MzSpectrum, min_probability: Option<f64>) -> MzSpectrum {

        let probability_cutoff = min_probability.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &spectrum.mz);

        let mut filtered_mz = Vec::new();
        let mut filtered_intensity = Vec::new();

        // zip mz and intensity with transmission probability and filter out all mz values with transmission probability 0.001
        for (i, (mz, intensity)) in spectrum.mz.iter().zip(spectrum.intensity.iter()).enumerate() {
            if transmission_probability[i] > probability_cutoff {
                filtered_mz.push(*mz);
                filtered_intensity.push(*intensity* transmission_probability[i]);
            }
        }

        MzSpectrum {
            mz: filtered_mz,
            intensity: filtered_intensity,
        }
    }

    fn transmit_ion(&self, frame_ids: Vec<i32>, scan_ids: Vec<i32>, spec: MzSpectrum, min_proba: Option<f64>) -> Vec<Vec<MzSpectrum>> {

        let mut result: Vec<Vec<MzSpectrum>> = Vec::new();

        for frame_id in frame_ids.iter() {
            let mut frame_result: Vec<MzSpectrum> = Vec::new();
            for scan_id in scan_ids.iter() {
                let transmitted_spectrum = self.transmit_spectrum(*frame_id, *scan_id, spec.clone(), min_proba);
                frame_result.push(transmitted_spectrum);
            }
            result.push(frame_result);
        }


        result
    }

    fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64, min_proba: Option<f64>) -> bool {
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &vec![mz]);
        transmission_probability[0] > probability_cutoff
    }

    fn any_transmitted(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>, min_proba: Option<f64>) -> bool {
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, mz);
        transmission_probability.iter().any(|&p| p > probability_cutoff)
    }

    fn transmit_tims_frame(&self, frame: &TimsFrame, min_probability: Option<f64>) -> TimsFrame {
        let spectra = frame.to_tims_spectra();
        let mut filtered_spectra = Vec::new();

        for mut spectrum in spectra {
            let filtered_spectrum = self.transmit_spectrum(frame.frame_id, spectrum.scan, spectrum.spectrum.mz_spectrum, min_probability);
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

    fn isotopes_transmitted(&self, frame_id: i32, scan_id: i32, mz_mono: f64, isotopic_envelope: &Vec<f64>, min_probability: Option<f64>) -> (f64, Vec<(f64, f64)>) {

        let probability_cutoff = min_probability.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &isotopic_envelope);
        let mut result: Vec<(f64, f64)> = Vec::new();

        for (mz, p) in isotopic_envelope.iter().zip(transmission_probability.iter()) {
            if *p > probability_cutoff {
                result.push((*mz - mz_mono, *p));
            }
        }

        (mz_mono, result)
    }
}

#[derive(Clone, Debug)]
pub struct TimsTransmissionDIA {
    frame_to_window_group: HashMap<i32, i32>,
    window_group_settings: HashMap<(i32, i32), (f64, f64)>,
    k: f64,
}

impl TimsTransmissionDIA {
    pub fn new(
        frame: Vec<i32>,
        frame_window_group: Vec<i32>,

        window_group: Vec<i32>,
        scan_start: Vec<i32>,
        scan_end: Vec<i32>,
        isolation_mz: Vec<f64>,
        isolation_width: Vec<f64>,
        k: Option<f64>,
    ) -> Self {
        // hashmap from frame to window group
        let frame_to_window_group = frame.iter().zip(frame_window_group.iter()).map(|(&f, &wg)| (f, wg)).collect::<HashMap<i32, i32>>();
        let mut window_group_settings: HashMap<(i32, i32), (f64, f64)> = HashMap::new();

        for (index, &wg) in window_group.iter().enumerate() {
            let scan_start = scan_start[index];
            let scan_end = scan_end[index];
            let isolation_mz = isolation_mz[index];
            let isolation_width = isolation_width[index];

            let value = (isolation_mz, isolation_width);

            for scan in scan_start..scan_end + 1 {
                let key = (wg, scan);
                window_group_settings.insert(key, value);
            }
        }

        Self {
            frame_to_window_group,
            window_group_settings,
            k: k.unwrap_or(1.0),
        }
    }

    pub fn frame_to_window_group(&self, frame_id: i32) -> i32 {
        let window_group = self.frame_to_window_group.get(&frame_id);
        match window_group {
            Some(&wg) => wg,
            None => -1,
        }
    }

    pub fn get_setting(&self, window_group: i32, scan_id: i32) -> Option<&(f64, f64)> {
        let setting = self.window_group_settings.get(&(window_group, scan_id));
        match setting {
            Some(s) => Some(s),
            None => None,
        }
    }

    // check if a frame is a precursor frame
    pub fn is_precursor(&self, frame_id: i32) -> bool {
        // if frame id is in the hashmap, it is not a precursor frame
        match self.frame_to_window_group.contains_key(&frame_id) {
            true => false,
            false => true,
        }
    }
}

impl IonTransmission for TimsTransmissionDIA {
    fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>) -> Vec<f64> {

        let setting = self.get_setting(self.frame_to_window_group(frame_id), scan_id);

        match setting {
            Some((isolation_mz, isolation_width)) => {
                apply_transmission(*isolation_mz, *isolation_width, self.k, mz.clone())
            },
            None => vec![1.0; mz.len()],
        }
    }
}