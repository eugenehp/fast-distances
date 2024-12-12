use ndarray::ArrayView1;
use num::Float;

/// Computes the cosine similarity between two vectors `x` and `y`.
///
/// The cosine similarity is defined as:
///
/// ..math::
///     \text{cosine}(x, y) = 1 - \frac{\sum x_i \cdot y_i}{\sqrt{\sum x_i^2} \cdot \sqrt{\sum y_i^2}}
///
/// If either vector has a norm of zero, the function will return `1.0` if one of the vectors is zero and `0.0` if both are zero.
///
/// # Arguments
///
/// * `x` - A 1D array representing the first vector.
/// * `y` - A 1D array representing the second vector.
///
/// # Returns
/// * A float representing the cosine similarity between the two vectors.
pub fn cosine<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float,
{
    let mut result = T::zero();
    let mut norm_x = T::zero();
    let mut norm_y = T::zero();

    for i in 0..x.len() {
        result = result + x[i] * y[i];
        norm_x = norm_x + x[i] * x[i];
        norm_y = norm_y + y[i] * y[i];
    }

    if norm_x.is_zero() && norm_y.is_zero() {
        T::zero()
    } else if norm_x.is_zero() || norm_y.is_zero() {
        T::one()
    } else {
        T::one() - (result / (norm_x.sqrt() * norm_y.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_cosine_basic_f32() {
        // Test with simple vectors using f32
        let x = arr1(&[1.0_f32, 2.0, 3.0]);
        let y = arr1(&[4.0_f32, 5.0, 6.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(
            result,
            1.0 - (32.0_f32 / (14.0_f32.sqrt() * 77.0_f32.sqrt()))
        );
    }

    #[test]
    fn test_cosine_zero_norm_f32() {
        // Test with a zero vector using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[1.0_f32, 2.0, 3.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(result, 1.0_f32);
    }

    #[test]
    fn test_cosine_zero_both_norm_f32() {
        // Test with two zero vectors using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[0.0_f32, 0.0, 0.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_cosine_basic_f64() {
        // Test with simple vectors using f64
        let x = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = arr1(&[4.0_f64, 5.0, 6.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(
            result,
            1.0 - (32.0_f64 / (14.0_f64.sqrt() * 77.0_f64.sqrt()))
        );
    }

    #[test]
    fn test_cosine_zero_norm_f64() {
        // Test with a zero vector using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[1.0_f64, 2.0, 3.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(result, 1.0_f64);
    }

    #[test]
    fn test_cosine_zero_both_norm_f64() {
        // Test with two zero vectors using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[0.0_f64, 0.0, 0.0]);
        let result = cosine(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }
}
