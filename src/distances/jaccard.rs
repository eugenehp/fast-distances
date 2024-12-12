extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Jaccard similarity between two binary vectors.
///
/// The Jaccard similarity is a measure of the similarity between two sets, defined as:
///
/// ..math::
///    J(x, y) = \frac{|x \cap y|}{|x \cup y|}
///
/// In this case, the vectors `x` and `y` are treated as binary vectors, where a non-zero value is considered as `True` and zero is considered as `False`.
///
/// The function counts the number of non-zero elements in `x` and `y`, the number of positions where both `x` and `y` are non-zero, and then calculates the similarity based on the formula:
///
/// ..math::
///    J(x, y) = \frac{(\text{number of non-zero positions in both x and y})}{(\text{number of non-zero positions in either x or y})}
///
/// If there are no non-zero elements in either vector, the function returns a similarity of `0.0`.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first vector.
/// * `y` - A 1D array (view) of values representing the second vector.
///
/// # Returns
/// A f64 value representing the Jaccard similarity.
pub fn jaccard(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut num_non_zero = 0.0;
    let mut num_equal = 0.0;

    for i in 0..x.len() {
        let x_true = x[i] != 0.0;
        let y_true = y[i] != 0.0;
        num_non_zero += if x_true || y_true { 1.0 } else { 0.0 };
        num_equal += if x_true && y_true { 1.0 } else { 0.0 };
    }

    if num_non_zero == 0.0 {
        0.0
    } else {
        (num_non_zero - num_equal) / num_non_zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_jaccard_basic() {
        let x = arr1(&[1.0, 0.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 0.0]);

        let similarity = jaccard(&x.view(), &y.view());
        let expected_similarity = 0.6666666666666666;
        assert_eq!(similarity, expected_similarity);
    }

    #[test]
    fn test_jaccard_identical_vectors() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = jaccard(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // All elements match
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let x = arr1(&[1.0, 0.0, 0.0]);
        let y = arr1(&[0.0, 0.0, 1.0]);

        let similarity = jaccard(&x.view(), &y.view());
        assert_eq!(similarity, 1.0); // No overlap, no common non-zero elements
    }

    #[test]
    fn test_jaccard_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let similarity = jaccard(&x.view(), &y.view());
        assert_eq!(similarity, 0.0); // Empty vectors, no non-zero elements
    }

    #[test]
    fn test_jaccard_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 1.0, 1.0]);

        let similarity = jaccard(&x.view(), &y.view());
        assert_eq!(similarity, 1.0); // x is zero vector, y is all ones
    }
}
