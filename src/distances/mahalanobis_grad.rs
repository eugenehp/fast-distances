use ndarray::{Array1, Array2, ArrayView1};
use num::Float;

use crate::utils::identity_matrix;

/// Computes the Mahalanobis distance and its gradient with respect to `x`
/// using the inverse covariance matrix `vinv`.
///
/// The Mahalanobis distance is a measure of the distance between a point and a distribution,
/// scaled by the inverse of the covariance matrix. The gradient of the Mahalanobis distance
/// is computed with respect to `x` and returned alongside the distance.
///
/// The formula for the Mahalanobis distance is:
///
/// ..math::
///     D(x, y) = \sqrt{ (x - y)^T \cdot V^{-1} \cdot (x - y) }
///
/// where `V^{-1}` is the inverse covariance matrix, and `(x - y)` is the difference between the vectors `x` and `y`.
///
/// The gradient of the Mahalanobis distance with respect to `x` is:
///
/// ..math::
///     \nabla_x D(x, y) = \frac{2 \cdot V^{-1} \cdot (x - y)}{\sqrt{(x - y)^T \cdot V^{-1} \cdot (x - y)}}
///
/// # Parameters:
/// - `x`: A reference to an `ArrayView1<T>` representing the first vector.
/// - `y`: A reference to an `ArrayView1<T>` representing the second vector.
/// - `vinv`: An optional reference to an `ArrayView2<T>` representing the inverse covariance matrix.
///           If `None`, it defaults to the identity matrix, effectively reducing the distance to Euclidean distance.
///
/// # Returns:
/// The Mahalanobis distance between `x` and `y`, and the gradient of the distance with respect to `x`
/// as an `Array1<T>`.
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
/// let (dist, grad) = mahalanobis_grad(&x.view(), &y.view(), Some(vinv));
/// println!("Mahalanobis Distance: {}", dist);
/// println!("Gradient: {:?}", grad);
/// ```
pub fn mahalanobis_grad<T>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
    vinv: Option<Array2<T>>,
) -> (T, Array1<T>)
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
    let mut grad_tmp = vec![T::zero(); x.len()];
    for i in 0..x.len() {
        let mut tmp = T::zero();
        for j in 0..x.len() {
            tmp = tmp + vinv[(i, j)] * diff[j];
            grad_tmp[i] = grad_tmp[i] + vinv[(i, j)] * diff[j];
        }
        result = result + tmp * diff[i];
    }

    let dist = result.sqrt();
    let grad: Array1<T> = grad_tmp
        .iter()
        .map(|&g| g / (T::from(1e-6).unwrap() + dist))
        .collect();

    (dist, Array1::from(grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ndarray::arr2;

    #[test]
    fn test_mahalanobis_grad_default_identity() {
        // Test when vinv is the identity matrix (Euclidean distance).
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let (dist, grad) = mahalanobis_grad(&x.view(), &y.view(), None);
        let expected_dist = ((3.0_f64.powi(2) + 3.0_f64.powi(2) + 3.0_f64.powi(2)) as f64).sqrt();
        let expected_grad = arr1(&[
            1.0 * (x[0] - y[0]) / (expected_dist + 1e-6),
            1.0 * (x[1] - y[1]) / (expected_dist + 1e-6),
            1.0 * (x[2] - y[2]) / (expected_dist + 1e-6),
        ]);

        assert!((dist - expected_dist).abs() < 1e-6);
        assert!((grad - expected_grad).sum().abs() < 1e-6);
    }

    #[test]
    fn test_mahalanobis_grad_with_vinv() {
        // Test with a specific inverse covariance matrix.
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let vinv = arr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]);

        let (dist, grad) = mahalanobis_grad(&x.view(), &y.view(), Some(vinv));
        let expected_dist = 6.708203932499369;
        let expected_grad = arr1(&[
            -0.6708202932499517,
            -0.8944270576666024,
            -0.6708202932499517,
        ]);

        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_mahalanobis_grad_edge_case() {
        // Test with identical vectors (distance should be 0 and gradient should be 0).
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let (dist, grad) = mahalanobis_grad(&x.view(), &y.view(), None);
        assert_eq!(dist, 0.0);
        assert_eq!(grad.sum(), 0.0);
    }
}
