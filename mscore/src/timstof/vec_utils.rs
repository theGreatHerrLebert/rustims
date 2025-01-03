pub fn find_sparse_local_maxima_mask(
    indices: &Vec<u32>,
    values: &Vec<f64>,
    window: u32,
) -> Vec<bool> {
    let mut local_maxima: Vec<bool> = vec![true; indices.len()];
    for (index, sparse_index) in indices.iter().enumerate() {
        let current_intensity: f64 = values[index];
        for (_next_index, next_sparse_index) in
            indices[index + 1..].iter().enumerate()
        {
            let next_index: usize = _next_index + index + 1;
            let next_value: f64 = values[next_index];
            if (next_sparse_index - sparse_index) <= window {
                if current_intensity < next_value {
                    local_maxima[index] = false
                } else {
                    local_maxima[next_index] = false
                }
            } else {
                break;
            }
        }
    }
    local_maxima
}

pub fn filter_with_mask<T: Copy>(vec: &Vec<T>, mask: &Vec<bool>) -> Vec<T> {
    (0..vec.len())
        .filter(|&x| mask[x])
        .map(|x| vec[x])
        .collect()
}