extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Kulsinski similarity between two binary vectors.
///
/// The Kulsinski similarity is defined as:
///
/// ..math::
///    S(x, y) = \frac{|x \neq y| - |x \cap y| + N}{|x \neq y| + N}
///
/// Where:
/// - `|x \neq y|` is the number of positions where the vectors differ (one is 1, the other is 0).
/// - `|x \cap y|` is the number of positions where both vectors are 1.
/// - `N` is the length of the vectors (the total number of elements).
///
/// This similarity metric is commonly used in binary data and treats the vectors as sets of 1's and 0's.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Kulsinski similarity.
pub fn kulsinski(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
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
        (num_not_equal - num_true_true + x.len() as f64) / (num_not_equal + x.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_kulsinski_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = kulsinski(&x.view(), &y.view());
        let expected_similarity = 0.8;
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_kulsinski_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = kulsinski(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors should return similarity of 0
    }

    #[test]
    fn test_kulsinski_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = kulsinski(&x.view(), &y.view());
        let expected_similarity = (3.0 - 0.0 + 3.0) / (3.0 + 3.0); // (num_not_equal - num_true_true + N) / (num_not_equal + N)
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_kulsinski_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = kulsinski(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Empty vectors should return 0 similarity
    }

    #[test]
    fn test_kulsinski_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = kulsinski(&x.view(), &y.view());
        let expected_similarity = (3.0 - 0.0 + 3.0) / (3.0 + 3.0); // (num_not_equal - num_true_true + N) / (num_not_equal + N)
        assert_eq!(similarity, expected_similarity);
    }
}
