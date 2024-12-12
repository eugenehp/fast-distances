use ndarray::{Array1, ArrayView1};
use num::Float;

/// Chebyshev or l-infinity distance with gradient.
///
/// Computes the Chebyshev distance (l-infinity norm) between two vectors `x` and `y`
/// and returns the distance along with the gradient of the distance with respect to `x`.
///
/// # Arguments
///
/// * `x` - A 1D array view representing the first vector.
/// * `y` - A 1D array view representing the second vector.
///
/// # Returns
///
/// A tuple containing:
/// - The Chebyshev distance between `x` and `y`.
/// - The gradient of the distance with respect to `x`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::chebyshev_grad;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let (distance, grad) = chebyshev_grad(x.view(), y.view());
///
/// assert_eq!(distance, 3.0);
/// assert_eq!(grad, arr1(&[-1.0, 0.0, 0.0]));
/// ```
pub fn chebyshev_grad<T: Float + num::Signed>(
    x: ArrayView1<'_, T>,
    y: ArrayView1<'_, T>,
) -> (T, Array1<T>) {
    assert_eq!(
        x.len(),
        y.len(),
        "Vectors x and y must have the same length"
    );

    let mut result = T::zero();
    let mut max_i = 0;

    for i in 0..x.len() {
        let v = (x[i] - y[i]).abs();
        if v > result {
            result = v;
            max_i = i;
        }
    }

    let mut grad = Array1::zeros(x.len());
    if result != T::zero() {
        grad[max_i] = (x[max_i] - y[max_i]).signum();
    }

    (result, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_chebyshev_grad() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let (distance, grad) = chebyshev_grad(x.view(), y.view());

        assert_eq!(distance, 3.0);
        assert_eq!(grad, arr1(&[-1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_chebyshev_grad_same_vectors() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);
        let (distance, grad) = chebyshev_grad(x.view(), y.view());

        assert_eq!(distance, 0.0);
        assert_eq!(grad, arr1(&[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_chebyshev_grad_mixed_signs() {
        let x = arr1(&[-1.0, 2.0, -3.0]);
        let y = arr1(&[1.0, -2.0, 3.0]);
        let (distance, grad) = chebyshev_grad(x.view(), y.view());

        assert_eq!(distance, 6.0);
        assert_eq!(grad, arr1(&[0.0, 0.0, -1.0]));
    }
}
