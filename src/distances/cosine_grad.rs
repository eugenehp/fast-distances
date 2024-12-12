use ndarray::{Array1, ArrayView1};
use num::Float;

/// Computes the cosine similarity and its gradient between two vectors `x` and `y`.
///
/// The cosine similarity is defined as:
///
/// ..math::
///     \text{cosine}(x, y) = 1 - \frac{\sum x_i \cdot y_i}{\sqrt{\sum x_i^2} \cdot \sqrt{\sum y_i^2}}
///
/// The gradient is computed with respect to each vector element. If either vector has a norm of zero, the gradient is zero.
///
/// # Arguments
///
/// * `x` - A 1D array representing the first vector.
/// * `y` - A 1D array representing the second vector.
///
/// # Returns
/// * A tuple containing:
///     - The cosine similarity between `x` and `y`.
///     - The gradient of the cosine similarity with respect to `x`.
pub fn cosine_grad<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (T, Array1<T>)
where
    T: Float,
{
    let mut result = T::zero();
    let mut norm_x = T::zero();
    let mut norm_y = T::zero();

    // Compute the dot product and the squared norms
    for i in 0..x.len() {
        result = result + x[i] * y[i];
        norm_x = norm_x + x[i] * x[i];
        norm_y = norm_y + y[i] * y[i];
    }

    // Initialize the gradient and distance (similarity)
    let (dist, grad) = if norm_x.is_zero() && norm_y.is_zero() {
        (T::zero(), Array1::<T>::zeros(x.dim()))
    } else if norm_x.is_zero() || norm_y.is_zero() {
        (T::one(), Array1::<T>::zeros(x.dim()))
    } else {
        let mut grad = Array1::<T>::zeros(x.dim());
        for i in 0..x.len() {
            grad[i] = -(x[i] * result - y[i] * norm_x)
                / (norm_x.powf(T::from(1.5).unwrap()) * norm_y.sqrt());
        }
        let dist = T::one() - (result / (norm_x.sqrt() * norm_y.sqrt()));
        (dist, grad)
    };

    (dist, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_cosine_grad_basic_f32() {
        // Test with simple vectors using f32
        let x = arr1(&[1.0_f32, 2.0, 3.0]);
        let y = arr1(&[4.0_f32, 5.0, 6.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        let expected_dist = 1.0 - (32.0_f32 / (14.0_f32.sqrt() * 77.0_f32.sqrt()));
        let expected_grad = arr1(&[
            -(1.0 * 32.0_f32 - 4.0 * 14.0_f32) / (14.0_f32.powf(1.5) * 77.0_f32.sqrt()),
            -(2.0 * 32.0_f32 - 5.0 * 14.0_f32) / (14.0_f32.powf(1.5) * 77.0_f32.sqrt()),
            -(3.0 * 32.0_f32 - 6.0 * 14.0_f32) / (14.0_f32.powf(1.5) * 77.0_f32.sqrt()),
        ]);
        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_cosine_grad_zero_norm_f32() {
        // Test with a zero vector using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[1.0_f32, 2.0, 3.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        assert_eq!(dist, 1.0_f32);
        assert_eq!(grad, arr1(&[0.0_f32, 0.0, 0.0]));
    }

    #[test]
    fn test_cosine_grad_zero_both_norm_f32() {
        // Test with two zero vectors using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[0.0_f32, 0.0, 0.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0_f32);
        assert_eq!(grad, arr1(&[0.0_f32, 0.0, 0.0]));
    }

    #[test]
    fn test_cosine_grad_basic_f64() {
        // Test with simple vectors using f64
        let x = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = arr1(&[4.0_f64, 5.0, 6.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        let expected_dist = 1.0 - (32.0_f64 / (14.0_f64.sqrt() * 77.0_f64.sqrt()));
        let expected_grad = arr1(&[
            -(1.0 * 32.0_f64 - 4.0 * 14.0_f64) / (14.0_f64.powf(1.5) * 77.0_f64.sqrt()),
            -(2.0 * 32.0_f64 - 5.0 * 14.0_f64) / (14.0_f64.powf(1.5) * 77.0_f64.sqrt()),
            -(3.0 * 32.0_f64 - 6.0 * 14.0_f64) / (14.0_f64.powf(1.5) * 77.0_f64.sqrt()),
        ]);
        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_cosine_grad_zero_norm_f64() {
        // Test with a zero vector using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[1.0_f64, 2.0, 3.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        assert_eq!(dist, 1.0_f64);
        assert_eq!(grad, arr1(&[0.0_f64, 0.0, 0.0]));
    }

    #[test]
    fn test_cosine_grad_zero_both_norm_f64() {
        // Test with two zero vectors using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[0.0_f64, 0.0, 0.0]);
        let (dist, grad) = cosine_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0_f64);
        assert_eq!(grad, arr1(&[0.0_f64, 0.0, 0.0]));
    }
}
