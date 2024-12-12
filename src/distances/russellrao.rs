extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Russell-Rao similarity between two binary vectors.
///
/// The Russell-Rao similarity is defined as:
///
/// ..math::
///    S(x, y) = \frac{N - |x \cap y|}{N}
///
/// Where:
/// - `|x \cap y|` is the number of positions where both vectors are `True` (non-zero).
/// - `N` is the length of the vectors.
///
/// The Russell-Rao similarity is a measure of the difference between two binary vectors, where `1` represents `True` and `0` represents `False`.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Russell-Rao similarity.
pub fn russell_rao(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut num_true_true = 0.0;

    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_true_true += if x_true && y_true { 1.0 } else { 0.0 };
    }

    // Check if both vectors are entirely non-zero
    let sum_x = x.iter().filter(|&&val| val != 0.0).count() as f64;
    let sum_y = y.iter().filter(|&&val| val != 0.0).count() as f64;

    if num_true_true == sum_x && num_true_true == sum_y {
        0.0 // If all non-zero elements match, return 0 similarity
    } else {
        (x.len() as f64 - num_true_true) / x.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_russell_rao_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = russell_rao(&x.view(), &y.view());
        let expected_similarity = (3.0 - 1.0) / 3.0; // (N - num_true_true) / N
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_russell_rao_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = russell_rao(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors should return 0 similarity
    }

    #[test]
    fn test_russell_rao_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = russell_rao(&x.view(), &y.view());
        let expected_similarity = (3.0 - 0.0) / 3.0; // (N - num_true_true) / N
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_russell_rao_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = russell_rao(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Empty vectors should return 0 similarity
    }

    #[test]
    fn test_russell_rao_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = russell_rao(&x.view(), &y.view());
        let expected_similarity = (3.0 - 0.0) / 3.0; // (N - num_true_true) / N
        assert_eq!(similarity, expected_similarity);
    }
}
