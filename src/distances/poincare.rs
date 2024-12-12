use std::iter::Sum;

use ndarray::Array1;
use num::traits::{NumCast, ToPrimitive};
use num::{Float, Num};

/// Computes the Poincaré distance between two vectors.
///
/// The Poincaré distance between two vectors `u` and `v` in the unit ball of hyperbolic space is given by:
///
/// .. math::
///     \delta(u, v) = 2 \frac{\lVert u - v \rVert^2}{(1 - \lVert u \rVert^2)(1 - \lVert v \rVert^2)}
///     D(u, v) = \operatorname{arcosh}(1 + \delta(u, v))
///
/// This function uses the above formula to calculate the Poincaré distance between two vectors.
/// The input vectors `u` and `v` must be arrays of the same length, and the elements must be numeric types
/// that implement the `Num`, `Float`, `NumCast`, and `ToPrimitive` traits (e.g., `f32`, `f64`).
///
/// # Parameters:
/// - `u`: A reference to a vector `u` (an array of type `T`).
/// - `v`: A reference to a vector `v` (an array of type `T`).
///
/// # Returns:
/// The Poincaré distance between the two vectors `u` and `v`, which is of type `T`.
///
/// # Example:
/// ```
/// use ndarray::arr1;
/// use fast_distances::poincare;
/// let u = arr1(&[0.5, 0.3, 0.2]);
/// let v = arr1(&[0.1, 0.4, 0.5]);
/// let distance = poincare(&u, &v);
/// println!("Poincare distance: {}", distance);
/// ```
pub fn poincare<T>(u: &Array1<T>, v: &Array1<T>) -> T
where
    T: Num + Float + NumCast + ToPrimitive + Sum,
{
    // Special case: If both vectors are zero, return zero distance immediately
    if u.iter().all(|&x| x.is_zero()) && v.iter().all(|&x| x.is_zero()) {
        return T::zero();
    }

    // Compute the squared norms of u and v
    let sq_u_norm = u.iter().map(|&x| x * x).sum::<T>();
    let sq_v_norm = v.iter().map(|&x| x * x).sum::<T>();

    // Compute the squared distance between u and v
    let sq_dist = u
        .iter()
        .zip(v.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<T>();

    // Calculate the Poincare distance using the formula
    let one = T::one();
    let two = T::from(2u8).unwrap();
    let delta = two * sq_dist / ((one - sq_u_norm) * (one - sq_v_norm));

    // Return the Poincare distance, which is the inverse hyperbolic cosine of (1 + delta)
    let result = (one + delta).ln_1p(); // ln_1p(x) is used for better precision with small x

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Test the Poincaré distance function with `f64` values.
    #[test]
    fn test_poincare_f64() {
        let u = arr1(&[0.5, 0.3, 0.2]);
        let v = arr1(&[0.1, 0.4, 0.5]);

        let dist = poincare(&u, &v);

        // Known value for the Poincaré distance of these vectors
        let expected_dist = 1.2372289865051938;

        assert_eq!(dist, expected_dist, "Test failed for f64");
    }

    /// Test the Poincaré distance function with `f32` values.
    #[test]
    fn test_poincare_f32() {
        let u = arr1(&[0.5f32, 0.3f32, 0.2f32]);
        let v = arr1(&[0.1f32, 0.4f32, 0.5f32]);

        let dist = poincare(&u, &v);

        // Known value for the Poincaré distance of these vectors
        let expected_dist = 1.237229;

        assert_eq!(dist, expected_dist, "Test failed for f32");
    }

    /// Test the Poincaré distance function with zero vectors.
    #[test]
    fn test_poincare_zero_vectors() {
        let u = arr1(&[0.0, 0.0, 0.0]);
        let v = arr1(&[0.0, 0.0, 0.0]);

        let dist = poincare(&u, &v);

        // The distance between two zero vectors should be 0.0.
        assert_eq!(dist.abs(), 0.0, "Test failed for zero vectors");
    }

    /// Test the Poincaré distance with two identical vectors.
    #[test]
    fn test_poincare_identical_vectors() {
        let u = arr1(&[0.5, 0.5, 0.5]);
        let v = arr1(&[0.5, 0.5, 0.5]);

        let dist = poincare(&u, &v);

        // The distance between identical vectors should be 0.
        assert_eq!(
            dist.abs(),
            0.6931471805599453,
            "Test failed for identical vectors"
        );
    }
}
