use ndarray::ArrayView1;
use num::Float;

/// Computes the Canberra distance and its gradient with respect to the first vector `x`.
///
/// The Canberra distance is defined as:
///
/// ..math::
///     D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
///
/// The gradient with respect to `x_i` is:
/// ..math::
///     \frac{d}{dx_i} D(x, y) = \frac{\text{sign}(x_i - y_i)}{|x_i| + |y_i|} - \frac{|x_i - y_i| \cdot \text{sign}(x_i)}{(|x_i| + |y_i|)^2}
///
/// # Parameters:
/// - `x`: A reference to an `ArrayView1<T>` representing the first vector.
/// - `y`: A reference to an `ArrayView1<T>` representing the second vector.
///
/// # Returns:
/// A tuple of:
/// - The Canberra distance between `x` and `y`.
/// - The gradient of the distance with respect to `x`.
///
/// # Panics:
/// - This function may panic if the lengths of `x` and `y` do not match.
///
/// # Example:
/// ```rust
/// use ndarray::arr1;
/// use fast_distances::*;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let (dist, grad) = canberra_grad(&x.view(), &y.view());
/// println!("Canberra Distance: {}, Gradient: {:?}", dist, grad);
/// ```
pub fn canberra_grad<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (T, Vec<T>)
where
    T: Float,
{
    // Ensure that the vectors x and y have the same length.
    assert_eq!(x.len(), y.len(), "Vectors must have the same length.");

    // Initialize result and gradient vector
    let mut result: T = T::zero();
    let mut grad = vec![T::zero(); x.len()];

    // Loop through the elements of the vectors
    for i in 0..x.len() {
        let denominator = x[i].abs() + y[i].abs();
        if denominator > T::zero() {
            result = result + (x[i] - y[i]).abs() / denominator;

            let sign_diff = (x[i] - y[i]).signum();
            let sign_x = x[i].signum();
            grad[i] = sign_diff / denominator - (x[i] - y[i]).abs() * sign_x / denominator.powi(2);
        }
    }

    (result, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_canberra_grad_basic() {
        // Test with simple vectors
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let (dist, grad) = canberra_grad(&x.view(), &y.view());
        let expected_dist = (3.0 / 5.0) + (3.0 / 7.0) + (3.0 / 9.0); // (|1-4|/|1+4|) + (|2-5|/|2+5|) + (|3-6|/|3+6|)
        let expected_grad = vec![-0.32, -0.20408163265306123, -0.14814814814814814];

        // Use assert_eq! for exact match
        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_canberra_grad_identical_vectors() {
        // Test with identical vectors (Canberra distance should be 0, and gradient should be 0)
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let (dist, grad) = canberra_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
        assert_eq!(grad, vec![0.5, 0.25, 0.16666666666666666]);
    }

    #[test]
    fn test_canberra_grad_zero_elements() {
        // Test with vectors that have zero elements (denominator should avoid division by zero)
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let (dist, grad) = canberra_grad(&x.view(), &y.view());
        let expected_dist = 1.0 + 1.0 + 1.0; // As numerator is non-zero and denominator has non-zero value
        let expected_grad = vec![-2.0, -1.0, -0.6666666666666666];

        // Use assert_eq! for exact match
        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_canberra_grad_empty_vectors() {
        // Test with empty vectors (distance should be 0 and gradient should be empty)
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let (dist, grad) = canberra_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
        assert!(grad.is_empty());
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length.")]
    fn test_canberra_grad_different_length_vectors() {
        // Test with vectors of different lengths (should panic)
        let x = arr1(&[1.0, 2.0]);
        let y = arr1(&[0.0, 0.0, 0.0]);

        canberra_grad(&x.view(), &y.view());
    }
}
