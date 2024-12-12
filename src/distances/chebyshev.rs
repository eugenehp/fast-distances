use ndarray::ArrayView1;
use num::Float;

/// Chebyshev or l-infinity distance.
///
/// Computes the Chebyshev distance (l-infinity norm) between two vectors `x` and `y`.
///
/// # Arguments
///
/// * `x` - A 1D array view representing the first vector.
/// * `y` - A 1D array view representing the second vector.
///
/// # Returns
///
/// The Chebyshev distance between `x` and `y`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::chebyshev;
///
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let distance = chebyshev(&x.view(), &y.view());
/// assert_eq!(distance, 3.0);
/// ```
pub fn chebyshev<T: Float>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "Vectors x and y must have the same length"
    );

    let mut result = T::zero();

    for i in 0..x.len() {
        result = result.max((x[i] - y[i]).abs());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_chebyshev() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let distance = chebyshev(&x.view(), &y.view());

        assert_eq!(distance, 3.0);
    }

    #[test]
    fn test_chebyshev_same_vectors() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);
        let distance = chebyshev(&x.view(), &y.view());

        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_chebyshev_mixed_signs() {
        let x = arr1(&[-1.0, 2.0, -3.0]);
        let y = arr1(&[1.0, -2.0, 3.0]);
        let distance = chebyshev(&x.view(), &y.view());

        assert_eq!(distance, 6.0);
    }
}
