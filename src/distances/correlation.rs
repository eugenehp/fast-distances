use ndarray::ArrayView1;
use num::Float;

/// Computes the Pearson correlation coefficient between two vectors `x` and `y`.
///
/// The Pearson correlation coefficient is defined as:
///
/// ..math::
///     r = 1 - \frac{\sum (x_i - \mu_x) \cdot (y_i - \mu_y)}{\sqrt{\sum (x_i - \mu_x)^2} \cdot \sqrt{\sum (y_i - \mu_y)^2}}
///
/// # Arguments
///
/// * `x` - A 1D array representing the first vector.
/// * `y` - A 1D array representing the second vector.
///
/// # Returns
/// The Pearson correlation coefficient between the vectors `x` and `y`. A value of 1.0 means the vectors are perfectly correlated,
/// and a value of 0.0 means no correlation.
pub fn correlation<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float,
{
    let mut mu_x = T::zero();
    let mut mu_y = T::zero();
    let mut norm_x = T::zero();
    let mut norm_y = T::zero();
    let mut dot_product = T::zero();

    // Compute the means (mu_x, mu_y)
    for i in 0..x.len() {
        mu_x = mu_x + x[i];
        mu_y = mu_y + y[i];
    }

    mu_x = mu_x / T::from(x.len()).unwrap();
    mu_y = mu_y / T::from(y.len()).unwrap();

    // Compute the dot product and norms
    for i in 0..x.len() {
        let shifted_x = x[i] - mu_x;
        let shifted_y = y[i] - mu_y;
        norm_x = norm_x + shifted_x * shifted_x;
        norm_y = norm_y + shifted_y * shifted_y;
        dot_product = dot_product + shifted_x * shifted_y;
    }

    if norm_x.is_zero() && norm_y.is_zero() {
        T::zero()
    } else if dot_product.is_zero() {
        T::one()
    } else {
        T::one() - (dot_product / (norm_x.sqrt() * norm_y.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_correlation_basic_f32() {
        // Test with simple vectors using f32
        let x = arr1(&[1.0_f32, 2.0, 3.0]);
        let y = arr1(&[4.0_f32, 5.0, 6.0]);
        let result = correlation(&x.view(), &y.view());
        let expected_result = -1.1920929e-7;
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_correlation_zero_norm_f32() {
        // Test with a zero vector using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[1.0_f32, 2.0, 3.0]);
        let result = correlation(&x.view(), &y.view());
        assert_eq!(result, 1.0_f32);
    }

    #[test]
    fn test_correlation_zero_both_norm_f32() {
        // Test with two zero vectors using f32
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[0.0_f32, 0.0, 0.0]);
        let result = correlation(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_correlation_basic_f64() {
        // Test with simple vectors using f64
        let x = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = arr1(&[4.0_f64, 5.0, 6.0]);
        let result = correlation(&x.view(), &y.view());
        let expected_result = 2.220446049250313e-16;
        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_correlation_zero_norm_f64() {
        // Test with a zero vector using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[1.0_f64, 2.0, 3.0]);
        let result = correlation(&x.view(), &y.view());
        assert_eq!(result, 1.0_f64);
    }

    #[test]
    fn test_correlation_zero_both_norm_f64() {
        // Test with two zero vectors using f64
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[0.0_f64, 0.0, 0.0]);
        let result = correlation(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }
}
