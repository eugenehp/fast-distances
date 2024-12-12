use std::iter::Sum;

use ndarray::Array1;
use num::traits::{NumCast, ToPrimitive};
use num::{Float, Num};

/// Computes the hyperboloid distance and gradient between two vectors `x` and `y`.
///
/// The hyperboloid distance between two vectors `x` and `y` is defined as:
///
/// .. math::
///     s = \sqrt{1 + \lVert x \rVert^2}, \quad t = \sqrt{1 + \lVert y \rVert^2}
///     B = s \cdot t - x \cdot y
///     \text{grad\_coeff} = \frac{1}{\sqrt{B - 1} \cdot \sqrt{B + 1}}
///     \text{grad} = \text{grad\_coeff} \cdot \left( \frac{x_i t}{s} - y_i \right)
///
/// # Parameters:
/// - `x`: A reference to a vector `x` (an array of type `T`).
/// - `y`: A reference to a vector `y` (an array of type `T`).
///
/// # Returns:
/// - The hyperboloid distance and its gradient as a tuple.
///
/// # Example:
/// ```
/// use ndarray::arr1;
/// use fast_distances::*;
///
/// let x = arr1(&[0.5, 0.3, 0.2]);
/// let y = arr1(&[0.1, 0.4, 0.5]);
/// let (distance, gradient) = hyperboloid_grad(&x, &y);
/// println!("Hyperboloid distance: {}, Gradient: {:?}", distance, gradient);
/// ```
pub fn hyperboloid_grad<T>(x: &Array1<T>, y: &Array1<T>) -> (T, Array1<T>)
where
    T: Num + Float + NumCast + ToPrimitive + Sum,
{
    // Calculate the norms and compute s and t
    let s = (T::one() + x.iter().map(|&xi| xi * xi).sum::<T>()).sqrt();
    let t = (T::one() + y.iter().map(|&yi| yi * yi).sum::<T>()).sqrt();

    // Compute B
    let mut b = s * t;
    for i in 0..x.len() {
        b = b - x[i] * y[i];
    }

    // Ensure that B > 1 to avoid invalid values in sqrt
    if b <= T::one() {
        b = T::one() + T::from(1e-8).unwrap();
    }

    // Calculate the gradient coefficient
    let grad_coeff = T::one() / ((b - T::one()).sqrt() * (b + T::one()).sqrt());

    // Initialize the gradient array
    let mut grad = Array1::<T>::zeros(x.len());

    // Compute the gradient
    for i in 0..x.len() {
        grad[i] = grad_coeff * ((x[i] * t) / s - y[i]);
    }

    // Return the hyperboloid distance and gradient
    (b.acosh(), grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Test the hyperboloid gradient function with `f64` values.
    #[test]
    fn test_hyperboloid_grad_f64() {
        let x = arr1(&[0.5, 0.3, 0.2]);
        let y = arr1(&[0.1, 0.4, 0.5]);

        let (dist, grad) = hyperboloid_grad(&x, &y);

        let expected_dist = 0.5042620265600418;
        let expected_grad = arr1(&[
            0.7742725795637821,
            -0.18193978944576575,
            -0.5649719537764168,
        ]);

        assert_eq!(
            dist, expected_dist,
            "Test failed for f64: Distance mismatch"
        );
        assert_eq!(
            grad, expected_grad,
            "Test failed for f64: Gradient mismatch"
        );
    }

    /// Test the hyperboloid gradient function with `f32` values.
    #[test]
    fn test_hyperboloid_grad_f32() {
        let x = arr1(&[0.5f32, 0.3f32, 0.2f32]);
        let y = arr1(&[0.1f32, 0.4f32, 0.5f32]);

        let (dist, grad) = hyperboloid_grad(&x, &y);

        let expected_dist = 0.50426185;
        let expected_grad = arr1(&[0.7742728, -0.18193986, -0.56497204]);

        assert_eq!(
            dist, expected_dist,
            "Test failed for f32: Distance mismatch"
        );
        assert_eq!(
            grad, expected_grad,
            "Test failed for f32: Gradient mismatch"
        );
    }

    /// Test the hyperboloid gradient function with zero vectors.
    #[test]
    fn test_hyperboloid_grad_zero_vectors() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 0.0]);

        let (dist, grad) = hyperboloid_grad(&x, &y);

        // The distance should not be 0 because of the way the function is designed.
        assert!(
            dist > 0.0,
            "Test failed for zero vectors: Distance mismatch"
        );
        assert!(
            grad.iter().all(|&g| g.abs() < 1e-6),
            "Test failed for zero vectors: Gradient mismatch"
        );
    }

    /// Test the hyperboloid gradient with two identical vectors.
    #[test]
    fn test_hyperboloid_grad_identical_vectors() {
        let x = arr1(&[0.5, 0.5, 0.5]);
        let y = arr1(&[0.5, 0.5, 0.5]);

        let (dist, grad) = hyperboloid_grad(&x, &y);

        // The distance between identical vectors should be 0.
        assert!(
            dist.abs() < 1e-6,
            "Test failed for identical vectors: Distance mismatch"
        );
        assert!(
            grad.iter().all(|&g| g.abs() < 1e-6),
            "Test failed for identical vectors: Gradient mismatch"
        );
    }
}
