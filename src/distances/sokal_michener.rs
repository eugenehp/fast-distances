extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Sokal-Michener similarity between two binary vectors.
///
/// The Sokal-Michener similarity is defined as:
///
/// ..math::
///    S(x, y) = \frac{2 \times |x \neq y|}{N + |x \neq y|}
///
/// Where:
/// - `|x \neq y|` is the number of positions where `x` and `y` are not equal.
/// - `N` is the length of the vectors.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Sokal-Michener similarity.
pub fn sokal_michener(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
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
    fn test_sokal_michener_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = sokal_michener(&x.view(), &y.view());
        let expected_similarity = (2.0 * 2.0) / (3.0 + 2.0); // (2 * num_not_equal) / (N + num_not_equal)
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_sokal_michener_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = sokal_michener(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors should return 0 similarity
    }

    #[test]
    fn test_sokal_michener_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = sokal_michener(&x.view(), &y.view());
        let expected_similarity = 0.8;
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_sokal_michener_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = sokal_michener(&x.view(), &y.view());
        assert!(similarity.is_nan()); // Empty vectors should return 0 similarity
    }

    #[test]
    fn test_sokal_michener_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = sokal_michener(&x.view(), &y.view());
        let expected_similarity = (2.0 * 3.0) / (3.0 + 3.0); // (2 * num_not_equal) / (N + num_not_equal)
        assert_eq!(similarity, expected_similarity);
    }
}
