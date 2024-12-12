extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Rogers-Tanimoto similarity between two binary vectors.
///
/// The Rogers-Tanimoto similarity is defined as:
///
/// ..math::
///    S(x, y) = \frac{2 |x \neq y|}{N + |x \neq y|}
///
/// Where:
/// - `|x \neq y|` is the number of positions where the vectors differ (one is 1, the other is 0).
/// - `N` is the length of the vectors (the total number of elements).
///
/// This similarity metric is used in binary data, where `0` represents `False` and any non-zero value represents `True`.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Rogers-Tanimoto similarity.
pub fn rogers_tanimoto(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut num_not_equal = 0.0;

    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_not_equal += if x_true != y_true { 1.0 } else { 0.0 };
    }

    (2.0 * num_not_equal) / (x.len() as f64 + num_not_equal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_rogers_tanimoto_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = rogers_tanimoto(&x.view(), &y.view());
        let expected_similarity = (2.0 * 2.0) / (3.0 + 2.0); // (2 * num_not_equal) / (N + num_not_equal)
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_rogers_tanimoto_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = rogers_tanimoto(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors should return 0 similarity
    }

    #[test]
    fn test_rogers_tanimoto_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = rogers_tanimoto(&x.view(), &y.view());
        let expected_similarity = 0.8;
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_rogers_tanimoto_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = rogers_tanimoto(&x.view(), &y.view());
        assert!(similarity.is_nan()); // Empty vectors should return 0 similarity
    }

    #[test]
    fn test_rogers_tanimoto_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = rogers_tanimoto(&x.view(), &y.view());
        let expected_similarity = (2.0 * 3.0) / (3.0 + 3.0); // (2 * num_not_equal) / (N + num_not_equal)
        assert_eq!(similarity, expected_similarity);
    }
}
