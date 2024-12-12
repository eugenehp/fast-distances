extern crate ndarray;

use ndarray::ArrayView1;

/// Computes the Bray-Curtis dissimilarity between two vectors.
///
/// The Bray-Curtis dissimilarity is a measure of dissimilarity between two samples, defined as:
///
/// ..math::
///    D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i |x_i + y_i|}
///
/// The result will be 0.0 if the denominator is zero.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first vector.
/// * `y` - A 1D array (view) of values representing the second vector.
///
/// # Returns
/// A floating-point value representing the Bray-Curtis dissimilarity between `x` and `y`.
pub fn bray_curtis(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        numerator += (x[i] - y[i]).abs();
        denominator += (x[i] + y[i]).abs();
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_bray_curtis_basic() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let result = bray_curtis(&x.view(), &y.view());
        let expected = (3.0 + 3.0 + 3.0) / (5.0 + 7.0 + 9.0); // (|1-4| + |2-5| + |3-6|) / (|1+4| + |2+5| + |3+6|)
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bray_curtis_identical_vectors() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let result = bray_curtis(&x.view(), &y.view());
        assert_eq!(result, 0.0); // Bray-Curtis dissimilarity is 0 for identical vectors
    }

    #[test]
    fn test_bray_curtis_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let result = bray_curtis(&x.view(), &y.view());
        let expected = (1.0 + 2.0 + 3.0) / (1.0 + 2.0 + 3.0); // (|0-1| + |0-2| + |0-3|) / (|0+1| + |0+2| + |0+3|)
        assert_eq!(result, expected);
    }

    #[test]
    fn test_bray_curtis_empty_vectors() {
        // Test with empty vectors, should return 0.0 as there's no data to compare
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let result = bray_curtis(&x.view(), &y.view());
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_bray_curtis_zero_denominator() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[-1.0, -1.0, -1.0]);

        let result = bray_curtis(&x.view(), &y.view());
        // Both vectors are opposites with the same absolute sum, so denominator is 0 and the result should be 0.0
        assert_eq!(result, 0.0);
    }
}
