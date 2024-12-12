use ndarray::{arr1, Array1, ArrayView1};
use num_traits::{Float, One, Zero};

/// Computes the standardised Euclidean distance between two vectors.
///
/// The standardised Euclidean distance takes into account the variance (or standard deviation)
/// of each feature, which is provided by the `sigma` parameter. If `sigma` is not provided,
/// it defaults to a vector of ones, effectively reducing the calculation to the regular
/// Euclidean distance.
///
/// # Arguments
///
/// * `x` - A reference to a 1-dimensional array view of type `T`.
/// * `y` - Another reference to a 1-dimensional array view of type `T`.
/// * `sigma` - An optional 1-dimensional array of type `T` representing the standard deviations
///             or variances for each feature. If not provided, defaults to a vector of ones.
///
/// # Returns
///
/// The standardised Euclidean distance between `x` and `y`, of type `T`.
///
/// # Panics
///
/// This function will panic if:
/// - The input arrays `x` and `y` do not have the same length.
/// - The `sigma` array (if provided) does not have the same length as `x` and `y`.
pub fn standardised_euclidean<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    sigma: Option<Array1<T>>,
) -> T
where
    T: Float + One + Zero,
{
    assert_eq!(x.len(), y.len());

    let len = x.len();
    let sigma_view = match sigma {
        Some(s) => s,
        None => arr1(&vec![T::one(); len]),
    };

    let mut result = T::zero();
    for i in 0..len {
        result = result + ((x[i] - y[i]) * (x[i] - y[i])) / sigma_view[i];
    }

    result.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_standardised_euclidean_with_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let sigma = arr1(&[1.0, 1.0, 1.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), Some(sigma));
        assert_eq!(distance, 5.196152422706632);
    }

    #[test]
    fn test_standardised_euclidean_without_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), None);
        assert_eq!(distance, 5.196152422706632);
    }

    #[test]
    fn test_standardised_euclidean_with_different_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let sigma = arr1(&[2.0, 2.0, 2.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), Some(sigma));
        assert_eq!(distance, 3.6742346141747673);
    }

    #[test]
    fn test_standardised_euclidean_with_zero_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let sigma = arr1(&[0.0, 1.0, 1.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), Some(sigma));
        assert_eq!(distance, f64::INFINITY)
    }

    #[test]
    fn test_standardised_euclidean_with_all_zero_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let sigma = arr1(&[0.0, 0.0, 0.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), Some(sigma));
        assert_eq!(distance, f64::INFINITY)
    }

    #[test]
    fn test_standardised_euclidean_with_negative_sigma() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let sigma = arr1(&[-1.0, 1.0, 1.0]);

        let distance = standardised_euclidean(&x.view(), &y.view(), Some(sigma));
        assert_eq!(distance, 3.0)
    }

    #[test]
    fn test_standardised_euclidean_with_f32() {
        let x = arr1(&[1.0f32, 2.0f32, 3.0f32]);
        let y = arr1(&[4.0f32, 5.0f32, 6.0f32]);

        let distance = standardised_euclidean(&x.view(), &y.view(), None);
        assert_eq!(distance, 5.196152)
    }
}
