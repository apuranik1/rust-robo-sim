/// Implementation of a Kalman filter
use ndarray::{Array1, Array2};
use ndarray_linalg::{error::LinalgError, Inverse};

/// A Gaussian distribution
/// cov may be singular
#[derive(Debug, Clone)]
pub struct GaussianBelief {
    mean: Array1<f64>,
    cov: Array2<f64>,
}

impl GaussianBelief {
    pub fn new(mean: Array1<f64>, cov: Array2<f64>) -> Self {
        GaussianBelief { mean, cov }
    }

    /// Compute an affine transformation ax + b
    pub fn affine_transform(&self, a: &Array2<f64>, b: &Array1<f64>) -> Self {
        let mean = a.dot(&self.mean) + b;
        let cov = a.dot(&self.cov).dot(&a.t());
        GaussianBelief { mean, cov }
    }

    /// Condition on an observation of the form cx = d
    /// d is itself a Gaussian belief
    /// Returns LinalgError if the prior covariance of cx is degenerate.
    pub fn condition(&self, c: &Array2<f64>, d: &GaussianBelief) -> Result<Self, LinalgError> {
        // in information form, we get
        // precision = self.cov()^-1 + c' d.cov()^-1 c
        // we can invert the above using Woodbury
        // info = c' d.cov()^-1 d.mean() + self.cov()^-1 self.mu
        let dim = self.mean.dim();
        let obs_cov = &d.cov + c.dot(&self.cov).dot(&c.t());
        let obs_res = &d.mean - c.dot(&self.mean);
        let obs_precision = obs_cov.inv()?;
        let gain = self.cov.dot(&c.t()).dot(&obs_precision);
        let post_mean = &self.mean + gain.dot(&obs_res);
        let post_cov = (Array2::eye(dim) - gain.dot(c)).dot(&self.cov);
        Ok(GaussianBelief {
            mean: post_mean,
            cov: post_cov,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use expect_test::expect;
    use ndarray::Array1;

    #[test]
    fn test_basic_update() {
        let mean: Array1<f64> = Array1::zeros(2);
        let cov: Array2<f64> = Array2::eye(2);
        let prior = GaussianBelief::new(mean, cov);
        // observe x + y = 1 with variance 1
        let emission = Array2::ones((1, 2));
        let observation = GaussianBelief::new(Array1::ones(1), Array2::ones((1, 1)));
        let post_1 = prior.condition(&emission, &observation).unwrap();
        let expected_mean = expect![[r#"
            [0.3333333333333333, 0.3333333333333333], shape=[2], strides=[1], layout=CFcf (0xf), const ndim=1
        "#]];
        let expected_cov = expect![[r#"
            [[0.6666666666666667, -0.3333333333333333],
             [-0.3333333333333333, 0.6666666666666667]], shape=[2, 2], strides=[2, 1], layout=Cc (0x5), const ndim=2
        "#]];
        expected_mean.assert_debug_eq(&post_1.mean);
        expected_cov.assert_debug_eq(&post_1.cov);
    }
}
