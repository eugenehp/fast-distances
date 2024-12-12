use ndarray::{Array2, ArrayView1};
use num::Float;

use crate::utils::identity_matrix;

/// Computes the Mahalanobis distance between two vectors `x` and `y` using the inverse covariance matrix `vinv`.
///
/// The Mahalanobis distance is a measure of the distance between a point and a distribution,
/// scaled by the inverse of the covariance matrix. It accounts for correlations between variables
/// and standardizes the differences according to the covariance structure.
///
/// The formula for the Mahalanobis distance is:
///
/// ..math::
///     D(x, y) = \sqrt{ (x - y)^T \cdot V^{-1} \cdot (x - y) }
///
/// where `V^{-1}` is the inverse covariance matrix, and `(x - y)` is the difference between the vectors `x` and `y`.
///
/// # Parameters:
/// - `x`: A reference to an `ArrayView1<T>` representing the first vector.
/// - `y`: A reference to an `ArrayView1<T>` representing the second vector.
/// - `vinv`: An optional reference to an `ArrayView2<T>` representing the inverse covariance matrix.
///           If `None`, it defaults to the identity matrix, effectively reducing the distance to Euclidean distance.
///
/// # Returns:
/// The Mahalanobis distance between `x` and `y` as a scalar of type `T`.
///
/// # Panics:
/// - This function may panic if the lengths of `x` and `y` do not match, or if the dimensions of `vinv` do not match the length of `x` or `y`.
///
/// # Example:
/// ```rust
/// use ndarray::{arr1, arr2};
/// use fast_distances::*;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let vinv = arr2(&[
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0]
/// ]);
/// let dist = mahalanobis(&x.view(), &y.view(), Some(vinv));
/// println!("Mahalanobis Distance: {}", dist);
/// ```
pub fn mahalanobis<T>(x: &ArrayView1<T>, y: &ArrayView1<T>, vinv: Option<Array2<T>>) -> T
where
    T: Float,
{
    // Default to identity matrix if vinv is None using the identity_matrix function
    let vinv = vinv.unwrap_or_else(|| {
        identity_matrix(x.len()) // Use the identity matrix if vinv is None
    });

    // Compute the difference (x - y)
    let mut diff = vec![T::zero(); x.len()];
    for i in 0..x.len() {
        diff[i] = x[i] - y[i];
    }

    // Compute the Mahalanobis distance
    let mut result = T::zero();
    for i in 0..x.len() {
        let mut tmp = T::zero();
        for j in 0..x.len() {
            tmp = tmp + vinv[(i, j)] * diff[j];
        }
        result = result + tmp * diff[i];
    }

    result.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ndarray::arr2;

    #[test]
    fn test_mahalanobis_default_identity() {
        // Test when vinv is the identity matrix (Euclidean distance).
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let result = mahalanobis(&x.view(), &y.view(), None);
        let expected = ((3.0_f64.powi(2) + 3.0_f64.powi(2) + 3.0_f64.powi(2)) as f64).sqrt();

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_mahalanobis_with_vinv() {
        // Test with a specific inverse covariance matrix.
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let vinv = arr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]);

        let result = mahalanobis(&x.view(), &y.view(), Some(vinv));
        let expected = 6.708203932499369;

        assert_eq!(result, expected);
    }

    #[test]
    fn test_mahalanobis_edge_case() {
        // Test with identical vectors (distance should be 0).
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let result = mahalanobis(&x.view(), &y.view(), None);
        assert_eq!(result, 0.0);
    }
}
