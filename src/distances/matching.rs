extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Matching similarity between two binary vectors.
///
/// The Matching similarity is the proportion of positions where the two binary vectors
/// differ in their values. The vectors are treated as binary, where non-zero values are
/// considered `True` and zero values are considered `False`.
///
/// The formula is:
///
/// ..math::
///    M(x, y) = \frac{|\{x_i \neq y_i\}|}{n}
///
/// where n is the number of elements in the vectors.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first binary vector.
/// * `y` - A 1D array (view) of values representing the second binary vector.
///
/// # Returns
/// A f64 value representing the Matching similarity.
pub fn matching(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut num_not_equal = 0.0;

    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_not_equal += if x_true != y_true { 1.0 } else { 0.0 };
    }

    num_not_equal / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_matching_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = matching(&x.view(), &y.view());
        let expected_similarity = 2.0 / 3.0; // Two positions are not equal: (x[1] != y[1]) and (x[2] != y[2])
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_matching_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = matching(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // All elements are equal, so no mismatches
    }

    #[test]
    fn test_matching_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = matching(&x.view(), &y.view());
        assert_eq!(similarity, 0.6666666666666666); // All elements are mismatched
    }

    #[test]
    fn test_matching_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = matching(&x.view(), &y.view());
        assert!(similarity.is_nan()); // Empty vectors, no mismatches
    }

    #[test]
    fn test_matching_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = matching(&x.view(), &y.view());
        assert_eq!(similarity, 1.0); // x is zero vector, y is all ones, all elements mismatch
    }
}
