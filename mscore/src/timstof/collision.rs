use std::collections::HashMap;

pub trait TimsTofCollisionEnergy {
    fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64;
}

pub struct TimsTofCollisionEnergyDIA {
    frame_to_window_group: HashMap<i32, i32>,
    window_group_settings: HashMap<(i32, i32), f64>,
}

impl TimsTofCollisionEnergyDIA {
    pub fn new(
        frame: Vec<i32>,
        frame_window_group: Vec<i32>,
        window_group: Vec<i32>,
        scan_start: Vec<i32>,
        scan_end: Vec<i32>,
        collision_energy: Vec<f64>,
    ) -> Self {
        // hashmap from frame to window group
        let frame_to_window_group = frame.iter().zip(frame_window_group.iter()).map(|(&f, &wg)| (f, wg)).collect::<HashMap<i32, i32>>();
        let mut window_group_settings: HashMap<(i32, i32), f64> = HashMap::new();

        for (index, &wg) in window_group.iter().enumerate() {
            let scan_start = scan_start[index];
            let scan_end = scan_end[index];
            let collision_energy = collision_energy[index];

            for scan in scan_start..scan_end + 1 {
                let key = (wg, scan);
                window_group_settings.insert(key, collision_energy);
            }
        }

        Self {
            frame_to_window_group,
            window_group_settings,
        }
    }
}

impl TimsTofCollisionEnergy for TimsTofCollisionEnergyDIA {
    fn get_collision_energy(&self, frame_id: i32, scan_id: i32) -> f64 {
        let window_group = self.frame_to_window_group.get(&frame_id);
        match window_group {
            Some(&wg) => {
                let setting = self.window_group_settings.get(&(wg, scan_id));
                match setting {
                    Some(&s) => s,
                    None => 0.0,
                }
            },
            None => 0.0,
        }
    }
}