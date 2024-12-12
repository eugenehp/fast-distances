use ndarray::ArrayView1;
use num::Float;

/// Minkowski distance.
///
/// Computes the Minkowski distance of order `p` between two vectors `x` and `y`.
///
/// # Arguments
///
/// * `x` - A 1D array view representing the first vector.
/// * `y` - A 1D array view representing the second vector.
/// * `p` - The order of the Minkowski distance. For p=1, it is equivalent to Manhattan distance;
///         for p=2, it is Euclidean distance; and for p=infinity, it is Chebyshev distance.
///
/// # Returns
///
/// The Minkowski distance between `x` and `y`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::minkowski;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let distance = minkowski(&x.view(), &y.view(), 2.0);
/// assert_eq!(distance, (3_f64.powi(2) * 3.0).sqrt());
/// ```
pub fn minkowski<T: Float>(x: &ArrayView1<T>, y: &ArrayView1<T>, p: T) -> T {
    assert_eq!(
        x.len(),
        y.len(),
        "Vectors x and y must have the same length"
    );

    let mut result = T::zero();

    for i in 0..x.len() {
        result = result + (x[i] - y[i]).abs().powf(p);
    }

    result.powf(T::one() / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr1;

    #[test]
    fn test_minkowski_euclidean() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let distance = minkowski(&x.view(), &y.view(), 2.0);
        assert_abs_diff_eq!(distance, (3_f64.powi(2) * 3.0).sqrt());
    }

    #[test]
    fn test_minkowski_manhattan() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let distance = minkowski(&x.view(), &y.view(), 1.0);
        assert_abs_diff_eq!(distance, 9.0);
    }

    #[test]
    fn test_minkowski_chebyshev() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let distance = minkowski(&x.view(), &y.view(), f64::INFINITY);
        assert_abs_diff_eq!(distance, 1.0);
    }

    #[test]
    fn test_minkowski_p_3() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let distance = minkowski(&x.view(), &y.view(), 3.0);
        assert_abs_diff_eq!(distance, (3_f64.powi(3) * 3.0).cbrt(), epsilon = 1.0e-12);
    }
}
