use ndarray::ArrayView1;
use num::Float;

/// Computes the Hellinger distance between two vectors `x` and `y`.
///
/// The Hellinger distance is defined as:
///
/// ..math::
///     H(x, y) = \sqrt{1 - \frac{\sum \sqrt{x_i y_i}}{\sqrt{\sum x_i \cdot \sum y_i}}}
///
/// where the sum is over the vector elements.
///
/// # Arguments
///
/// * `x` - A 1D array representing the first vector.
/// * `y` - A 1D array representing the second vector.
///
/// # Returns
/// The Hellinger distance between the vectors `x` and `y`. A value of 0 means the vectors are identical,
/// and a value of 1 means the vectors are completely different.
pub fn hellinger<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float,
{
    let mut result = T::zero();
    let mut l1_norm_x = T::zero();
    let mut l1_norm_y = T::zero();

    // Compute the sum of sqrt(x_i * y_i) and the L1 norms of x and y
    for i in 0..x.len() {
        result = result + (x[i] * y[i]).sqrt();
        l1_norm_x = l1_norm_x + x[i];
        l1_norm_y = l1_norm_y + y[i];
    }

    if l1_norm_x.is_zero() && l1_norm_y.is_zero() {
        T::zero()
    } else if l1_norm_x.is_zero() || l1_norm_y.is_zero() {
        T::one()
    } else {
        T::one() - (result / (l1_norm_x * l1_norm_y).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_hellinger_basic_f32() {
        // Test with simple vectors using f32
        let x = arr1(&[1.0_f32, 2.0, 3.0]);
        let y = arr1(&[4.0_f32, 5.0, 6.0]);
        let result = hellinger(&x.view(), &y.view());
        let expected_result = 0.008634508;
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_hellinger_zero_norm_f32() {
        // Test with a zero vector using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[1.0_f32, 2.0, 3.0]);
        let result = hellinger(&x.view(), &y.view());
        assert_eq!(result, 1.0_f32);
    }

    #[test]
    fn test_hellinger_zero_both_norm_f32() {
        // Test with two zero vectors using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[0.0_f32, 0.0, 0.0]);
        let result = hellinger(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_hellinger_basic_f64() {
        // Test with simple vectors using f64
        let x = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = arr1(&[4.0_f64, 5.0, 6.0]);
        let result = hellinger(&x.view(), &y.view());
        let expected_result = 0.008634560488816612;
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_hellinger_zero_norm_f64() {
        // Test with a zero vector using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[1.0_f64, 2.0, 3.0]);
        let result = hellinger(&x.view(), &y.view());
        assert_eq!(result, 1.0_f64);
    }

    #[test]
    fn test_hellinger_zero_both_norm_f64() {
        // Test with two zero vectors using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[0.0_f64, 0.0, 0.0]);
        let result = hellinger(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }
}
