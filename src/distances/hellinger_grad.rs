use ndarray::{Array1, ArrayView1};
use num::Float;

/// Computes the Hellinger gradient and the Hellinger distance between two vectors `x` and `y`.
///
/// The Hellinger gradient is calculated along with the distance. If either of the L1 norms of `x` or `y` is zero,
/// the distance is set to 1 (maximum dissimilarity), and the gradient is set to zero.
///
/// # Arguments
///
/// * `x` - A 1D array representing the first vector.
/// * `y` - A 1D array representing the second vector.
///
/// # Returns
/// A tuple containing:
/// - The Hellinger distance between the vectors `x` and `y`.
/// - The gradient of the Hellinger distance with respect to the vector `x`.
pub fn hellinger_grad<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (T, Array1<T>)
where
    T: Float,
{
    let mut result = T::zero();
    let mut l1_norm_x = T::zero();
    let mut l1_norm_y = T::zero();

    let mut grad_term = Array1::<T>::zeros(x.len()); // Correctly specify the array type

    // Compute the gradient term (sqrt(x_i * y_i)) and the L1 norms of x and y
    for i in 0..x.len() {
        grad_term[i] = (x[i] * y[i]).sqrt();
        result = result + grad_term[i];
        l1_norm_x = l1_norm_x + x[i];
        l1_norm_y = l1_norm_y + y[i];
    }

    let dist;
    let grad;

    if l1_norm_x.is_zero() && l1_norm_y.is_zero() {
        dist = T::zero();
        grad = Array1::<T>::zeros(x.len());
    } else if l1_norm_x.is_zero() || l1_norm_y.is_zero() {
        dist = T::one();
        grad = Array1::<T>::zeros(x.len());
    } else {
        let dist_denom = (l1_norm_x * l1_norm_y).sqrt();
        dist = (T::one() - result / dist_denom).sqrt();

        let grad_denom = T::from(2.0).unwrap() * dist;
        let grad_numer_const = (l1_norm_y * result) / (T::from(2.0).unwrap() * dist_denom.powi(3));

        // Use `zip` to perform element-wise operations between `y` and `grad_term`
        grad = grad_term
            .into_iter()
            .zip(y.into_iter())
            .map(|(grad_term_val, y_val)| grad_numer_const - (*y_val / grad_term_val * dist_denom))
            .map(|val| val / grad_denom)
            .collect::<Array1<T>>();
    }

    (dist, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_hellinger_grad_basic_f32() {
        let x = arr1(&[1.0_f32, 2.0, 3.0]);
        let y = arr1(&[4.0_f32, 5.0, 6.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());

        let expected_dist = 0.092922054;
        assert_eq!(dist, expected_dist);

        let expected_grad = arr1(&[-101.649994, -80.26827, -71.7472]);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_hellinger_grad_zero_norm_f32() {
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[1.0_f32, 2.0, 3.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());
        assert_eq!(dist, 1.0_f32);
        assert_eq!(grad, arr1(&[0.0_f32, 0.0, 0.0]));
    }

    #[test]
    fn test_hellinger_grad_zero_both_norm_f32() {
        let x = arr1(&[0.0_f32, 0.0, 0.0]);
        let y = arr1(&[0.0_f32, 0.0, 0.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0_f32);
        assert_eq!(grad, arr1(&[0.0_f32, 0.0, 0.0]));
    }

    #[test]
    fn test_hellinger_grad_basic_f64() {
        let x = arr1(&[1.0_f64, 2.0, 3.0]);
        let y = arr1(&[4.0_f64, 5.0, 6.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());

        let expected_dist = 0.09292233579079151;
        assert_eq!(dist, expected_dist);

        let expected_grad = arr1(&[-101.649684188505, -80.26803290308787, -71.74698077108695]);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_hellinger_grad_zero_norm_f64() {
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[1.0_f64, 2.0, 3.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());
        assert_eq!(dist, 1.0_f64);
        assert_eq!(grad, arr1(&[0.0_f64, 0.0, 0.0]));
    }

    #[test]
    fn test_hellinger_grad_zero_both_norm_f64() {
        let x = arr1(&[0.0_f64, 0.0, 0.0]);
        let y = arr1(&[0.0_f64, 0.0, 0.0]);
        let (dist, grad) = hellinger_grad(&x.view(), &y.view());
        assert_eq!(dist, 0.0_f64);
        assert_eq!(grad, arr1(&[0.0_f64, 0.0, 0.0]));
    }
}
