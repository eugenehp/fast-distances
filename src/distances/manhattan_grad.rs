use ndarray::{Array1, ArrayView1};
use num::Float;

/// Manhattan, taxicab, or l1 distance with gradient.
///
/// Computes the L1 distance between two vectors `x` and `y`, as well as the gradient of the distance
/// with respect to `x`.
///
/// # Arguments
///
/// * `x` - A 1D array view representing the first vector.
/// * `y` - A 1D array view representing the second vector.
///
/// # Returns
///
/// A tuple containing:
/// * The L1 distance between `x` and `y`.
/// * The gradient of the L1 distance with respect to `x`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::*;
///
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
///
/// let (distance, grad) = manhattan_grad(&x.view(), &y.view());
/// assert_eq!(distance, 9.0);
/// assert_eq!(grad, arr1(&[-1.0, -1.0, -1.0]));
/// ```
pub fn manhattan_grad<T: Float + num::Signed>(
    x: &ArrayView1<T>,
    y: &ArrayView1<T>,
) -> (T, Array1<T>) {
    assert_eq!(
        x.len(),
        y.len(),
        "Vectors x and y must have the same length"
    );

    let mut result = T::zero();
    let mut grad = Array1::<T>::zeros(x.dim());

    for i in 0..x.len() {
        let diff = x[i] - y[i];
        result = result + diff.abs();
        grad[i] = diff.signum();
    }

    (result, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_manhattan_grad() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        let (distance, grad) = manhattan_grad(&x.view(), &y.view());

        assert_eq!(distance, 9.0);
        assert_eq!(grad, arr1(&[-1.0, -1.0, -1.0]));
    }

    #[test]
    fn test_manhattan_grad_same_vectors() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);
        let (distance, grad) = manhattan_grad(&x.view(), &y.view());

        assert_eq!(distance, 0.0);
        assert_eq!(grad, arr1(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_manhattan_grad_mixed_signs() {
        let x = arr1(&[-1.0, 2.0, -3.0]);
        let y = arr1(&[1.0, -2.0, 3.0]);
        let (distance, grad) = manhattan_grad(&x.view(), &y.view());

        assert_eq!(distance, 12.0);
        assert_eq!(grad, arr1(&[-1.0, 1.0, -1.0]));
    }
}
