use std::f64::consts::E;

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