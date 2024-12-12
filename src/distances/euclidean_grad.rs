use ndarray::Array1;
use num::Float;

/// Computes the Euclidean distance and its gradient between two vectors.
///
/// The function calculates the Euclidean distance between two input vectors `x` and `y`
/// and returns both the distance and the gradient. The gradient indicates how much each
/// element in the input vectors contributes to the distance.
///
/// # Parameters
///
/// - **`x`:** An `Array1<T>` representing the first vector.
/// - **`y`:** An `Array1<T>` representing the second vector.
///
/// # Type Parameter
///
/// - **`T`:** A generic type that must implement the `Float` trait. This ensures
///   that the elements in vectors `x` and `y` can be used for arithmetic operations
///   involving floating-point numbers.
///
/// # Returns
///
/// A tuple containing:
/// 1. The Euclidean distance between the two input vectors, of type `T`.
/// 2. An `Array1<T>` representing the gradient. Each element in this array corresponds to the contribution of each element in the input vectors towards the Euclidean distance.
///
/// # Panics
///
/// - If the input arrays do not have the same length, the function will panic with an appropriate error message.
pub fn euclidean_grad<T>(x: &Array1<T>, y: &Array1<T>) -> (T, Vec<T>)
where
    T: Float,
{
    assert_eq!(x.len(), y.len(), "Input arrays must have the same length.");

    let mut result = T::zero();
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        result = result + diff * diff;
    }

    let distance = result.sqrt();
    let mut gradient = Vec::with_capacity(x.len());

    // Calculate the gradient
    for i in 0..x.len() {
        let grad = (x[i] - y[i]) / (T::from(1e-6).unwrap() + distance);
        gradient.push(grad);
    }

    (distance, gradient)
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*; // Import the function to be tested

    #[test]
    fn test_euclidean_grad_f64() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!(
            (dist - 5.196152422706632).abs() < 1e-6,
            "Distance is incorrect for f64."
        );
        assert!(
            (grad[0] - -0.5773502691896257).abs() < 1e-6,
            "Gradient[0] is incorrect for f64."
        );
        assert!(
            (grad[1] - -0.5773502691896257).abs() < 1e-6,
            "Gradient[1] is incorrect for f64."
        );
        assert!(
            (grad[2] - -0.5773502691896257).abs() < 1e-6,
            "Gradient[2] is incorrect for f64."
        );
    }

    #[test]
    fn test_euclidean_grad_f32() {
        let x = arr1(&[1.0f32, 2.0, 3.0]);
        let y = arr1(&[4.0f32, 5.0, 6.0]);

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!(
            (dist - 5.1961524).abs() < 1e-6,
            "Distance is incorrect for f32."
        );
        assert!(
            (grad[0] - -0.57735026).abs() < 1e-6,
            "Gradient[0] is incorrect for f32."
        );
        assert!(
            (grad[1] - -0.57735026).abs() < 1e-6,
            "Gradient[1] is incorrect for f32."
        );
        assert!(
            (grad[2] - -0.57735026).abs() < 1e-6,
            "Gradient[2] is incorrect for f32."
        );
    }

    #[test]
    fn test_euclidean_grad_zero_distance() {
        let x = arr1(&[1.0f64, 2.0, 3.0]);
        let y = arr1(&[1.0f64, 2.0, 3.0]);

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!(
            (dist - 0.0).abs() < 1e-6,
            "Distance should be 0 for identical vectors."
        );
        for &g in grad.iter() {
            assert!(
                (g - 0.0).abs() < 1e-6,
                "Gradient should be 0 for identical vectors."
            );
        }
    }

    #[test]
    #[should_panic(expected = "Input arrays must have the same length.")]
    fn test_euclidean_grad_different_lengths() {
        let x = arr1(&[1.0f64, 2.0]);
        let y = arr1(&[4.0f64, 5.0, 6.0]);
        euclidean_grad(&x, &y); // This should panic
    }
}
