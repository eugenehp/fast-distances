extern crate ndarray;
extern crate num;

use ndarray::ArrayView1;
use num::Float;

/// Computes the Yule's Q statistic (a measure of association between two binary variables).
///
/// The Yule's Q statistic is a measure of the association between two binary variables based on the counts
/// of true-positive, false-positive, true-negative, and false-negative outcomes. It is defined as:
///
/// .. math::
///     Q = \frac{2 \cdot (True\_Positive \cdot False\_Negative)}{(True\_Positive \cdot False\_Positive) + (True\_Negative \cdot False\_Negative)}
///
/// If either the number of false positives or false negatives is zero, the function returns `0.0`.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary variable.
/// * `y` - A 1D array (view) of values representing the second binary variable.
///
/// # Returns
/// A floating-point value representing the Yule's Q statistic.
///
/// # Panics
/// Panics if the input arrays are not of the same length.
pub fn yule<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float,
{
    if x.len() != y.len() {
        panic!("Input arrays must have the same length");
    }

    let mut num_true_true = T::zero();
    let mut num_true_false = T::zero();
    let mut num_false_true = T::zero();

    for i in 0..x.len() {
        let x_true = x[i] != T::zero();
        let y_true = y[i] != T::zero();

        if x_true && y_true {
            num_true_true = num_true_true + T::one();
        } else if x_true && !y_true {
            num_true_false = num_true_false + T::one();
        } else if !x_true && y_true {
            num_false_true = num_false_true + T::one();
        }
    }

    let num_false_false =
        T::from(x.len()).unwrap() - num_true_true - num_true_false - num_false_true;

    if num_true_false == T::zero() || num_false_true == T::zero() {
        return T::zero();
    }

    (T::from(2.0).unwrap() * num_true_false * num_false_true)
        / (num_true_true * num_false_false + num_true_false * num_false_true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_yule_basic_f32() {
        // Test with simple binary variables using f32
        let x = arr1(&[1.0_f32, 0.0, 1.0, 1.0]);
        let y = arr1(&[1.0_f32, 0.0, 0.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_yule_zero_false_f32() {
        // Test with no false positives using f32
        let x = arr1(&[1.0_f32, 1.0, 1.0, 0.0]);
        let y = arr1(&[1.0_f32, 1.0, 1.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_yule_equal_arrays_f32() {
        // Test with identical arrays using f32
        let x = arr1(&[1.0_f32, 1.0, 1.0, 1.0]);
        let y = arr1(&[1.0_f32, 1.0, 1.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f32);
    }

    #[test]
    fn test_yule_no_agreement_f32() {
        // Test with no agreement between the arrays using f32
        let x = arr1(&[1.0_f32, 0.0, 1.0, 0.0]);
        let y = arr1(&[0.0_f32, 1.0, 0.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 2.0_f32);
    }

    #[test]
    fn test_yule_different_lengths_f32() {
        // Test for invalid input lengths (should panic) using f32
        let x = arr1(&[1.0_f32, 0.0, 1.0, 0.0]);
        let y = arr1(&[1.0_f32, 0.0, 1.0]);
        let result = std::panic::catch_unwind(|| {
            yule(&x.view(), &y.view());
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_yule_basic_f64() {
        // Test with simple binary variables using f64
        let x = arr1(&[1.0_f64, 0.0, 1.0, 1.0]);
        let y = arr1(&[1.0_f64, 0.0, 0.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }

    #[test]
    fn test_yule_zero_false_f64() {
        // Test with no false positives using f64
        let x = arr1(&[1.0_f64, 1.0, 1.0, 0.0]);
        let y = arr1(&[1.0_f64, 1.0, 1.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }

    #[test]
    fn test_yule_equal_arrays_f64() {
        // Test with identical arrays using f64
        let x = arr1(&[1.0_f64, 1.0, 1.0, 1.0]);
        let y = arr1(&[1.0_f64, 1.0, 1.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 0.0_f64);
    }

    #[test]
    fn test_yule_no_agreement_f64() {
        // Test with no agreement between the arrays using f64
        let x = arr1(&[1.0_f64, 0.0, 1.0, 0.0]);
        let y = arr1(&[0.0_f64, 1.0, 0.0, 1.0]);
        let result = yule(&x.view(), &y.view());
        assert_eq!(result, 2.0_f64);
    }

    #[test]
    fn test_yule_different_lengths_f64() {
        // Test for invalid input lengths (should panic) using f64
        let x = arr1(&[1.0_f64, 0.0, 1.0, 0.0]);
        let y = arr1(&[1.0_f64, 0.0, 1.0]);
        let result = std::panic::catch_unwind(|| {
            yule(&x.view(), &y.view());
        });
        assert!(result.is_err());
    }
}
