extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Sokal-Sneath similarity between two binary vectors.
///
/// The Sokal-Sneath similarity is defined as:
///
/// ..math::
///    S(x, y) = \frac{|x \neq y|}{0.5 \times |x \land y| + |x \neq y|}
///
/// Where:
/// - `|x \neq y|` is the number of positions where `x` and `y` are not equal.
/// - `|x \land y|` is the number of positions where both `x` and `y` are true (non-zero).
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Sokal-Sneath similarity.
pub fn sokal_sneath(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
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
        num_not_equal / (0.5 * num_true_true + num_not_equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_sokal_sneath_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = sokal_sneath(&x.view(), &y.view());
        let num_true_true = 1.0; // only the first position is true for both
        let num_not_equal = 2.0; // the second and third positions are different
        let expected_similarity = num_not_equal / (0.5 * num_true_true + num_not_equal);
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_sokal_sneath_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = sokal_sneath(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Identical vectors should return 0 similarity
    }

    #[test]
    fn test_sokal_sneath_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = sokal_sneath(&x.view(), &y.view());
        let num_true_true = 0.0;
        let num_not_equal = 3.0; // all positions differ
        let expected_similarity = num_not_equal / (0.5 * num_true_true + num_not_equal);
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_sokal_sneath_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = sokal_sneath(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Empty vectors should return 0 similarity
    }

    #[test]
    fn test_sokal_sneath_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = sokal_sneath(&x.view(), &y.view());
        let num_true_true = 0.0;
        let num_not_equal = 3.0; // all positions differ
        let expected_similarity = num_not_equal / (0.5 * num_true_true + num_not_equal);
        assert_eq!(similarity, expected_similarity);
    }
}
