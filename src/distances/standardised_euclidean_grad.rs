use ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};

/// Euclidean distance standardised against a vector of standard deviations per coordinate with gradient.
///
/// # Arguments
///
/// * `x` - An array view for the first input vector.
/// * `y` - An array view for the second input vector.
/// * `sigma` - An optional array view for the standard deviations. If not provided, defaults to an array of ones.
///
/// # Returns
///
/// A tuple containing the distance and the gradient.
pub fn standardised_euclidean_grad<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    sigma: Option<Array1<T>>,
) -> (T, Array1<T>)
where
    T: Float + FromPrimitive,
{
    assert_eq!(x.len(), y.len());

    let sigma = match sigma {
        Some(s) => {
            assert_eq!(x.len(), s.len());
            s
        }
        None => Array1::<T>::ones(x.len()),
    };

    let mut result = T::zero();
    for i in 0..x.len() {
        let diff: T = x[i] - y[i];
        let s: T = sigma[i];
        result = result + (diff * diff) / s;
    }
    let d: T = result.sqrt();

    let mut grad = Array1::<T>::zeros(x.len());
    let epsilon = T::from_f64(1e-6).unwrap();
    for i in 0..x.len() {
        let diff: T = x[i] - y[i];
        let s: T = sigma[i];
        grad[i] = diff / (epsilon + d * s);
    }

    (d, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_standardised_euclidean_grad_with_sigma() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);
        let sigma = arr1(&[1.0f64, 1.0, 1.0]);

        let (d, grad) = standardised_euclidean_grad(&x.view(), &y.view(), Some(sigma));

        // Expected distance
        let expected_d = ((4.0 - 1.0).powi(2) + (5.0 - 2.0).powi(2) + (6.0 - 3.0).powi(2)).sqrt();
        assert!((d - expected_d).abs() < 1e-9);

        // Expected gradient
        let epsilon = 1e-6;
        let expected_grad_x0 = (1.0 - 4.0) / (epsilon + expected_d * 1.0);
        let expected_grad_x1 = (2.0 - 5.0) / (epsilon + expected_d * 1.0);
        let expected_grad_x2 = (3.0 - 6.0) / (epsilon + expected_d * 1.0);

        assert!((grad[0] - expected_grad_x0).abs() < 1e-9);
        assert!((grad[1] - expected_grad_x1).abs() < 1e-9);
        assert!((grad[2] - expected_grad_x2).abs() < 1e-9);
    }

    #[test]
    fn test_standardised_euclidean_grad_without_sigma() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);

        let (d, grad) = standardised_euclidean_grad(&x.view(), &y.view(), None);

        // Expected distance
        let expected_d = ((4.0 - 1.0).powi(2) + (5.0 - 2.0).powi(2) + (6.0 - 3.0).powi(2)).sqrt();
        assert!((d - expected_d).abs() < 1e-9);

        // Expected gradient
        let epsilon = 1e-6;
        let expected_grad_x0 = (1.0 - 4.0) / (epsilon + expected_d * 1.0);
        let expected_grad_x1 = (2.0 - 5.0) / (epsilon + expected_d * 1.0);
        let expected_grad_x2 = (3.0 - 6.0) / (epsilon + expected_d * 1.0);

        assert!((grad[0] - expected_grad_x0).abs() < 1e-9);
        assert!((grad[1] - expected_grad_x1).abs() < 1e-9);
        assert!((grad[2] - expected_grad_x2).abs() < 1e-9);
    }
}
