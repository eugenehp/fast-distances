extern crate ndarray;

use ndarray::{Array1, ArrayView1};

/// Computes the Bray-Curtis dissimilarity and its gradient between two vectors.
///
/// The Bray-Curtis dissimilarity is a measure of dissimilarity between two samples, defined as:
///
/// ..math::
///    D(x, y) = \frac{\sum_i |x_i - y_i|}{\sum_i |x_i + y_i|}
///
/// The gradient is calculated as:
///
/// ..math::
///    \nabla D(x, y) = \frac{\text{sign}(x_i - y_i) - D(x, y)}{|x_i + y_i|}
///
/// If the denominator is zero, the function returns a distance of 0.0 and a zero gradient.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first vector.
/// * `y` - A 1D array (view) of values representing the second vector.
///
/// # Returns
/// A tuple containing:
/// - The Bray-Curtis dissimilarity (f64).
/// - The gradient of the dissimilarity with respect to the first vector `x` (Array1<f64>).
pub fn bray_curtis_grad(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> (f64, Array1<f64>) {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..x.len() {
        numerator += (x[i] - y[i]).abs();
        denominator += (x[i] + y[i]).abs();
    }

    let (dist, grad) = if denominator > 0.0 {
        let dist = numerator / denominator;
        let mut grad = Array1::<f64>::zeros(x.len());
        for i in 0..x.len() {
            grad[i] = (x[i] - y[i]).signum() - dist;
            grad[i] /= denominator;
        }
        (dist, grad)
    } else {
        (0.0, Array1::<f64>::zeros(x.len()))
    };

    (dist, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_bray_curtis_grad_basic() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let (dist, grad) = bray_curtis_grad(&x.view(), &y.view());
        let expected_dist = (3.0 + 3.0 + 3.0) / (5.0 + 7.0 + 9.0); // (|1-4| + |2-5| + |3-6|) / (|1+4| + |2+5| + |3+6|)
        let expected_grad = arr1(&[
            -0.06802721088435375,
            -0.06802721088435375,
            -0.06802721088435375,
        ]);

        // Use assert_eq! for exact match on distance and entire gradient
        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_bray_curtis_grad_identical_vectors() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let (dist, grad) = bray_curtis_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
        assert_eq!(
            grad,
            arr1(&[
                0.08333333333333333,
                0.08333333333333333,
                0.08333333333333333
            ])
        );
    }

    #[test]
    fn test_bray_curtis_grad_zero_vector() {
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let (dist, grad) = bray_curtis_grad(&x.view(), &y.view());
        let expected_dist = (1.0 + 2.0 + 3.0) / (1.0 + 2.0 + 3.0);
        let expected_grad = arr1(&[
            -0.3333333333333333,
            -0.3333333333333333,
            -0.3333333333333333,
        ]);

        assert_eq!(dist, expected_dist);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_bray_curtis_grad_empty_vectors() {
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let (dist, grad) = bray_curtis_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
        assert_eq!(grad, arr1::<f64>(&[]));
    }

    #[test]
    fn test_bray_curtis_grad_zero_denominator() {
        let x = arr1(&[1.0, 1.0, 1.0]);
        let y = arr1(&[-1.0, -1.0, -1.0]);

        let (dist, grad) = bray_curtis_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
        assert_eq!(grad, arr1(&[0.0, 0.0, 0.0]));
    }
}
