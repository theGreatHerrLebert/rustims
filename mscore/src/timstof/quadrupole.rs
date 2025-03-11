use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64;
use std::f64::consts::E;
use itertools::izip;
use crate::data::spectrum::MzSpectrum;
use crate::simulation::annotation::{MzSpectrumAnnotated, TimsFrameAnnotated};
use crate::timstof::frame::TimsFrame;

/// Sigmoid step function for quadrupole selection simulation
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

    let up_start = midpoint - half_window - 0.5;
    let up_end = midpoint - half_window;
    let down_start = midpoint + half_window;
    let down_end = midpoint + half_window + 0.5;

    // take a vector of mz values to their transmission probability
    move |mz: Vec<f64>| -> Vec<f64> {
        smooth_step_up_down(&mz, up_start, up_end, down_start, down_end, k)
    }
}

/// Apply ion transmission function to mz values
///
/// Arguments:
///
/// * `midpoint` - center of the step
/// * `window_length` - length of the step
/// * `k` - steepness of the step
/// * `mz` - mz values
///
/// Returns:
///
/// * `Vec<f64>` - transmission probability for each mz value
///
/// # Examples
///
/// ```
/// use mscore::timstof::quadrupole::apply_transmission;
///
/// let mz = vec![100.0, 150.0, 170.0];
/// let transmission = apply_transmission(150.0, 50.0, 1.0, mz).iter().map(
/// |&x| (x * 100.0).round() / 100.0).collect::<Vec<f64>>();
/// assert_eq!(transmission, vec![0.0, 1.0, 1.0]);
/// ```
pub fn apply_transmission(midpoint: f64, window_length: f64, k: f64, mz: Vec<f64>) -> Vec<f64> {
    ion_transition_function_midpoint(midpoint, window_length, k)(mz)
}

pub trait IonTransmission {
    fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>) -> Vec<f64>;

    /// Transmit a spectrum given a frame id and scan id
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `spectrum` - MzSpectrum
    /// * `min_probability` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `MzSpectrum` - transmitted spectrum
    ///
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

    /// Transmit an annotated spectrum given a frame id and scan id
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `spectrum` - MzSpectrumAnnotated
    /// * `min_probability` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `MzSpectrumAnnotated` - transmitted spectrum
    ///
    fn transmit_annotated_spectrum(&self, frame_id: i32, scan_id: i32, spectrum: MzSpectrumAnnotated, min_probability: Option<f64>) -> MzSpectrumAnnotated {
        let probability_cutoff = min_probability.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &spectrum.mz);

        let mut filtered_mz = Vec::new();
        let mut filtered_intensity = Vec::new();
        let mut filtered_annotation = Vec::new();

        // zip mz and intensity with transmission probability and filter out all mz values with transmission probability 0.5
        for (i, (mz, intensity, annotation)) in izip!(spectrum.mz.iter(), spectrum.intensity.iter(), spectrum.annotations.iter()).enumerate() {
            if transmission_probability[i] > probability_cutoff {
                filtered_mz.push(*mz);
                filtered_intensity.push(*intensity* transmission_probability[i]);
                filtered_annotation.push(annotation.clone());
            }
        }

        MzSpectrumAnnotated {
            mz: filtered_mz,
            intensity: filtered_intensity,
            annotations: filtered_annotation,
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

    /// Get all ions in a frame that are transmitted
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `mz` - mz values
    /// * `min_proba` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `HashSet<usize>` - indices of transmitted mz values
    ///
    fn get_transmission_set(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>, min_proba: Option<f64>) -> HashSet<usize> {
        // go over enumerated mz and push all indices with transmission probability > min_proba to a set
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, mz);
        mz.iter().enumerate().filter(|&(i, _)| transmission_probability[i] > probability_cutoff).map(|(i, _)| i).collect()
    }

    /// Check if all mz values in a given collection are transmitted
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `mz` - mz values
    /// * `min_proba` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `bool` - true if all mz values are transmitted
    ///
    fn all_transmitted(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>, min_proba: Option<f64>) -> bool {
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, mz);
        transmission_probability.iter().all(|&p| p > probability_cutoff)
    }

    /// Check if a single mz value is transmitted
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `mz` - mz value
    /// * `min_proba` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `bool` - true if mz value is transmitted
    ///
    fn is_transmitted(&self, frame_id: i32, scan_id: i32, mz: f64, min_proba: Option<f64>) -> bool {
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, &vec![mz]);
        transmission_probability[0] > probability_cutoff
    }

    /// Check if any mz value is transmitted, can be used to check if one peak of isotopic envelope is transmitted
    ///
    /// Arguments:
    ///
    /// * `frame_id` - frame id
    /// * `scan_id` - scan id
    /// * `mz` - mz values
    /// * `min_proba` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `bool` - true if any mz value is transmitted
    ///
    fn any_transmitted(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>, min_proba: Option<f64>) -> bool {
        let probability_cutoff = min_proba.unwrap_or(0.5);
        let transmission_probability = self.apply_transmission(frame_id, scan_id, mz);
        transmission_probability.iter().any(|&p| p > probability_cutoff)
    }

    /// Transmit a frame given a diaPASEF transmission layout
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

    /// Transmit a frame given a diaPASEF transmission layout with annotations
    ///
    /// Arguments:
    ///
    /// * `frame` - TimsFrameAnnotated
    /// * `min_probability` - minimum probability for transmission
    ///
    /// Returns:
    ///
    /// * `TimsFrameAnnotated` - transmitted frame
    ///
    fn transmit_tims_frame_annotated(&self, frame: &TimsFrameAnnotated, min_probability: Option<f64>) -> TimsFrameAnnotated {
        let spectra = frame.to_tims_spectra_annotated();
        let mut filtered_spectra = Vec::new();

        for mut spectrum in spectra {
            let filtered_spectrum = self.transmit_annotated_spectrum(frame.frame_id, spectrum.scan as i32, spectrum.spectrum.clone(), min_probability);
            if filtered_spectrum.mz.len() > 0 {
                spectrum.spectrum = filtered_spectrum;
                filtered_spectra.push(spectrum);
            }
        }

        if  filtered_spectra.len() > 0 {
            TimsFrameAnnotated::from_tims_spectra_annotated(filtered_spectra)
        } else {
            TimsFrameAnnotated::new(
                frame.frame_id,
                frame.retention_time,
                frame.ms_type.clone(),
                vec![],
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
            k: k.unwrap_or(15.0),
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
        let is_precursor = self.is_precursor(frame_id);

        match setting {
            Some((isolation_mz, isolation_width)) => {
                apply_transmission(*isolation_mz, *isolation_width, self.k, mz.clone())
            },
            None => match is_precursor {
                true => vec![1.0; mz.len()],
                false => vec![0.0; mz.len()],
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PASEFMeta {
    pub frame: i32,
    pub scan_start: i32,
    pub scan_end: i32,
    pub isolation_mz: f64,
    pub isolation_width: f64,
    pub collision_energy: f64,
    pub precursor: i32,
}

impl PASEFMeta {
    pub fn new(frame: i32, scan_start: i32, scan_end: i32, isolation_mz: f64, isolation_width: f64, collision_energy: f64, precursor: i32) -> Self {
        Self {
            frame,
            scan_start,
            scan_end,
            isolation_mz,
            isolation_width,
            collision_energy,
            precursor,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimsTransmissionDDA {
    // frame id to corresponding pasef meta data
    pub pasef_meta: BTreeMap<i32, Vec<PASEFMeta>>,
    pub k: f64,
}

impl TimsTransmissionDDA {
    pub fn new(pasef_meta: Vec<PASEFMeta>, k: Option<f64>) -> Self {
        let mut pasef_map: BTreeMap<i32, Vec<PASEFMeta>> = BTreeMap::new();
        for meta in pasef_meta {
            let entry = pasef_map.entry(meta.frame).or_insert(Vec::new());
            entry.push(meta);
        }
        Self {
            pasef_meta: pasef_map,
            k: k.unwrap_or(15.0),
        }
    }

    pub fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> Option<f64> {
        let frame_meta = self.pasef_meta.get(&frame_id);
        match frame_meta {
            Some(meta) => {
                for m in meta {
                    if scan_id >= m.scan_start && scan_id <= m.scan_end {
                        return Some(m.collision_energy);
                    }
                }
                None
            },
            None => None,
        }
    }
}

impl IonTransmission for TimsTransmissionDDA {
    fn apply_transmission(&self, frame_id: i32, scan_id: i32, mz: &Vec<f64>) -> Vec<f64> {

        // get all selections for a frame, if frame is not in the PASEF metadata, it is a precursor frame and all ions are transmitted
        let meta = self.pasef_meta.get(&frame_id);

        match meta {
            Some(meta) => {
                let mut transmission = vec![0.0; mz.len()];

                for m in meta {
                    // check if scan id is in the range of the selection
                    if scan_id >= m.scan_start && scan_id <= m.scan_end {
                        // apply transmission function to mz values
                        let transmission_prob = apply_transmission(m.isolation_mz, m.isolation_width, self.k, mz.clone());
                        // make sure that the transmission probability is not lower than the previous one
                        for (i, p) in transmission_prob.iter().enumerate() {
                            transmission[i] = p.max(transmission[i]);
                        }
                    }
                }
                transmission
            },
            // if frame is not in the metadata, all ions are transmitted (precursor frame)
            None => vec![1.0; mz.len()],
        }
    }
}