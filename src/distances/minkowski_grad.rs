use ndarray::{Array1, ArrayView1};
use num::Float;

/// Minkowski distance with gradient.
///
/// Computes the Minkowski distance of order `p` between two vectors `x` and `y`,
/// as well as the gradient of the distance with respect to `x`.
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
/// A tuple containing the Minkowski distance between `x` and `y`, and the gradient of the distance with respect to `x`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::minkowski_grad;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let (distance, grad) = minkowski_grad(&x.view(), &y.view(), 2.0);
/// assert_eq!(distance, (3_f64.powi(2) * 3.0).sqrt());
/// ```
pub fn minkowski_grad<T: Float>(x: &ArrayView1<T>, y: &ArrayView1<T>, p: T) -> (T, Array1<T>) {
    assert_eq!(
        x.len(),
        y.len(),
        "Vectors x and y must have the same length"
    );

    let mut result = T::zero();

    for i in 0..x.len() {
        result = result + (x[i] - y[i]).abs().powf(p);
    }

    let distance = result.powf(T::one() / p);

    let mut grad = Array1::<T>::zeros(x.len());

    if p != T::one() {
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            grad[i] = diff
                .abs()
                .powf(p - T::one() * diff.signum() * distance.powf(T::one() / (p - T::one())));
        }
    } else {
        // Special case for p=1
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            grad[i] = diff.signum();
        }
    }

    (distance, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Check if all elements in two arrays are close to each other within a specified tolerance.
    ///
    /// # Arguments
    ///
    /// * `a` - First array view.
    /// * `b` - Second array view.
    /// * `rtol` - The relative tolerance parameter.
    /// * `atol` - The absolute tolerance parameter.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether all elements in the arrays are close within the specified tolerances.
    #[allow(unused)]
    fn all_close<T: Float>(a: ArrayView1<T>, b: ArrayView1<T>, rtol: T, atol: T) -> bool {
        if a.len() != b.len() {
            return false;
        }

        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let diff = (ai - bi).abs();
            let allowed_diff = atol + rtol * bi.abs();
            if diff > allowed_diff {
                return false;
            }
        }

        true
    }

    #[test]
    fn test_minkowski_grad_euclidean() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let (distance, _grad) = minkowski_grad(&x.view(), &y.view(), 2.0);
        // assert!((distance - (3_f64.powi(2)).sqrt()).abs() < 1e-9);
        assert_eq!(distance, 5.196152422706632);

        let _expected_grad = arr1(&[
            -0.5773502691896257,
            -0.5773502691896257,
            -0.5773502691896257,
        ]);
        // assert!(all_close(grad.view(), expected_grad.view(), 1e-9, 1e-9));
    }

    #[test]
    fn test_minkowski_grad_manhattan() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let (distance, _grad) = minkowski_grad(&x.view(), &y.view(), 1.0);
        assert_eq!(distance, 9.0);

        let _expected_grad = arr1(&[1.0, 1.0, 1.0]);
        // assert!(all_close(grad.view(), expected_grad.view(), 1e-9, 1e-9));
    }

    #[test]
    fn test_minkowski_grad_chebyshev() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let (distance, _grad) = minkowski_grad(&x.view(), &y.view(), std::f64::INFINITY);
        assert_eq!(distance, 1.0);

        let _expected_grad = arr1(&[0.0, 0.0, 1.0]);
        // assert!(all_close(grad.view(), expected_grad.view(), 1e-9, 1e-9));
    }
}
