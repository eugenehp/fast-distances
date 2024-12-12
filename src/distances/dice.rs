extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Dice coefficient between two binary vectors.
///
/// The Dice coefficient is a measure of similarity between two sets, defined as:
///
/// ..math::
///    D(x, y) = \frac{2 |x \cap y|}{|x| + |y|}
///
/// In this case, the vectors `x` and `y` are treated as binary vectors, where a non-zero value is considered as `True` and zero is considered as `False`.
/// The Dice coefficient is calculated as the ratio of the number of positions where both vectors have non-zero values, to the total number of positions where at least one of the vectors has a non-zero value.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Dice coefficient.
pub fn dice(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut num_true_true = 0.0;
    let mut num_not_equal = 0.0;

    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += if x_true && y_true { 1.0 } else { 0.0 };
        num_not_equal += if x_true != y_true { 1.0 } else { 0.0 };
    }

    if num_not_equal == 0.0 {
        0.0
    } else {
        num_not_equal / (2.0 * num_true_true + num_not_equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_dice_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = dice(&x.view(), &y.view());
        let expected_similarity = 0.5;
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_dice_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = dice(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors, no mismatches
    }

    #[test]
    fn test_dice_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = dice(&x.view(), &y.view());
        assert_eq!(similarity, 1.0); // No overlap between non-zero elements, so it returns 1
    }

    #[test]
    fn test_dice_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = dice(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Empty vectors, no mismatches
    }

    #[test]
    fn test_dice_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = dice(&x.view(), &y.view());
        assert_eq!(similarity, 1.0); // x is zero vector, y is all ones, all elements mismatch
    }
}
