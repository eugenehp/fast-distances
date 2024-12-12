use ndarray::ArrayView1;
use num::{Float, Zero};

/// Computes the Euclidean distance between two vectors.
///
/// # Arguments
///
/// * `x` - A 1-dimensional array view of type `T`.
/// * `y` - Another 1-dimensional array view of type `T`.
///
/// # Returns
///
/// The Euclidean distance between `x` and `y`, of type `T`.
///
/// # Panics
///
/// This function will panic if the input arrays do not have the same length.
pub fn euclidean<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float + Zero,
{
    assert_eq!(x.len(), y.len(), "Input arrays must have the same length.");

    let mut result = T::zero();
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        result = result + diff * diff;
    }

    result.sqrt()
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*; // Import the function to be tested

    #[test]
    fn test_euclidean_f64() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);

        let dist = euclidean(&x.view(), &y.view());
        assert!(
            (dist - 5.196152422706632).abs() < 1e-6,
            "Test failed for f64."
        );
    }

    #[test]
    fn test_euclidean_f32() {
        let x = arr1(&[1.0f32, 2.0, 3.0]);
        let y = arr1(&[4.0f32, 5.0, 6.0]);

        let dist = euclidean(&x.view(), &y.view());
        assert!((dist - 5.1961524).abs() < 1e-6, "Test failed for f32.");
    }

    #[test]
    fn test_euclidean_zero_distance() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[1.0f64, 2.0, 3.0]);

        let dist = euclidean(&x.view(), &y.view());
        assert!((dist - 0.0).abs() < 1e-6, "Test failed for zero distance.");
    }

    #[test]
    #[should_panic(expected = "Input arrays must have the same length.")]
    fn test_euclidean_different_lengths() {
        let x = arr1(&[1.0f64, 2.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);
        euclidean(&x.view(), &y.view()); // This should panic
    }
}
