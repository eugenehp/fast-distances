use ndarray::Array1;
use num::traits::{NumCast, ToPrimitive};
use num::{Float, Num};

/// Computes the weighted Minkowski distance between two vectors `x` and `y` with optional weights `w`
/// and a parameter `p` (defaulting to 2 for Euclidean distance).
///
/// The formula for the weighted Minkowski distance is:
///
/// .. math::
///     D(x, y) = \left( \sum_i w_i |x_i - y_i|^p \right)^{\frac{1}{p}}
///
/// # Parameters:
/// - `x`: A reference to a vector `x` (an array of type `T`).
/// - `y`: A reference to a vector `y` (an array of type `T`).
/// - `w`: A reference to a vector `w` of weights (an array of type `T`). If not provided, defaults to an array of ones.
/// - `p`: A parameter `p` for the Minkowski distance. Defaults to `2` for Euclidean distance.
///
/// # Returns:
/// The weighted Minkowski distance between `x` and `y`.
///
/// # Example:
/// ```
/// use fast_distances::weighted_minkowski;
/// let x = ndarray::Array1::from(vec![1.0, 2.0, 3.0]);
/// let y = ndarray::Array1::from(vec![4.0, 5.0, 6.0]);
/// let w = ndarray::Array1::from(vec![0.5, 0.5, 0.5]);
/// let dist = weighted_minkowski(&x, &y, Some(&w), 2.0);
/// println!("Weighted Minkowski distance: {}", dist);
/// ```
pub fn weighted_minkowski<T>(x: &Array1<T>, y: &Array1<T>, w: Option<&Array1<T>>, p: T) -> T
where
    T: Num + Float + NumCast + ToPrimitive,
{
    // Use weights w if provided, otherwise assume they are all 1.0
    let w = match w {
        Some(w) => w,
        None => &Array1::<T>::ones(x.len()), // Default weights of 1.0 for each dimension
    };

    // Compute the weighted Minkowski distance
    let mut result = T::zero();

    for i in 0..x.len() {
        result = result + w[i] * (x[i] - y[i]).abs().powf(p);
    }

    // Return the result raised to the power of 1/p
    result.powf(T::one() / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// Test the weighted Minkowski distance function with `f64` values.
    #[test]
    fn test_weighted_minkowski_f64() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let w = arr1(&[0.5, 0.5, 0.5]);

        let dist = weighted_minkowski(&x, &y, Some(&w), 2.0);

        // Known value for the weighted Minkowski distance of these vectors
        let expected_dist = 3.6742346141747673;

        // Use `assert_eq!` directly with floating point values and a small tolerance
        assert_eq!(dist.to_f64().unwrap(), expected_dist, "Test failed for f64");
    }

    /// Test the weighted Minkowski distance function with `f32` values.
    #[test]
    fn test_weighted_minkowski_f32() {
        let x = arr1(&[1.0f32, 2.0f32, 3.0f32]);
        let y = arr1(&[4.0f32, 5.0f32, 6.0f32]);
        let w = arr1(&[0.5f32, 0.5f32, 0.5f32]);

        let dist = weighted_minkowski(&x, &y, Some(&w), 2.0);

        // Known value for the weighted Minkowski distance of these vectors
        let expected_dist = 3.674234628677368;

        // Use `assert_eq!` directly with floating point values and a small tolerance
        assert_eq!(
            dist.to_f64().unwrap(),
            expected_dist.to_f64().unwrap(),
            "Test failed for f32"
        );
    }

    /// Test the weighted Minkowski distance function without weights.
    #[test]
    fn test_weighted_minkowski_no_weights() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let dist = weighted_minkowski(&x, &y, None, 2.0);

        // Known value for the Euclidean distance of these vectors
        let expected_dist = 5.196152422706632;

        // Use `assert_eq!` directly with floating point values and a small tolerance
        assert_eq!(dist.to_f64().unwrap(), expected_dist, "Test failed for f64");
    }

    /// Test the weighted Minkowski distance with `p` equal to 1 (Manhattan distance).
    #[test]
    fn test_weighted_minkowski_p1() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let w = arr1(&[0.5, 0.5, 0.5]);

        let dist = weighted_minkowski(&x, &y, Some(&w), 1.0);

        // Known value for the weighted Manhattan distance of these vectors
        let expected_dist = 4.5;

        // Use `assert_eq!` directly with floating point values and a small tolerance
        assert_eq!(dist.to_f64().unwrap(), expected_dist, "Test failed for f64");
    }
}
