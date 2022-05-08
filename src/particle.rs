/// Implementation of a particle filter
use rand::Rng;

pub fn resample<P: Clone>(particles: &[P], weights: &[f64]) -> Vec<P> {
    // unbiased n log n implementation of particle resampling
    let total_weight = weights.iter().sum();
    let n_samples = particles.len();
    if n_samples == 0 {
        return vec![];
    }
    let mut rng = rand::thread_rng();
    let mut draws: Vec<f64> = { 0..n_samples }
        .map(|_| rng.gen::<f64>() * total_weight)
        .collect();
    draws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut index = 0;
    let mut cum_weight = draws[0];
    let mut samples = Vec::with_capacity(n_samples);
    for draw in draws.iter() {
        // maintain invariant cum_weight >= draw
        // where cum_weight = weight sum from 0 to index inclusive
        while &cum_weight < draw {
            if index == n_samples - 1 {
                // weird precision edge case
                cum_weight = total_weight;
                break;
            } else {
                cum_weight += weights[index];
                index += 1;
            }
        }
        samples.push(particles[index].clone());
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doesnt_crash() {
        let particles: Vec<i64> = (0..5).collect();
        let weights = vec![1., 2., 3., 4., 5.];
        let sample = resample(&particles, &weights);
        assert_eq!(sample.len(), particles.len());
    }
}
