use num::Float;

pub fn euclidean_grad<T>(x: &[T], y: &[T]) -> (T, Vec<T>)
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
    use super::*; // Import the function to be tested

    #[test]
    fn test_euclidean_grad_f64() {
        let x = vec![1.0f64, 2.0, 3.0];
        let y = vec![4.0f64, 5.0, 6.0];

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!((dist - 5.196152422706632).abs() < 1e-6, "Distance is incorrect for f64.");
        assert!((grad[0] - -0.5773502691896257).abs() < 1e-6, "Gradient[0] is incorrect for f64.");
        assert!((grad[1] - -0.5773502691896257).abs() < 1e-6, "Gradient[1] is incorrect for f64.");
        assert!((grad[2] - -0.5773502691896257).abs() < 1e-6, "Gradient[2] is incorrect for f64.");
    }

    #[test]
    fn test_euclidean_grad_f32() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![4.0f32, 5.0, 6.0];

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!((dist - 5.1961524).abs() < 1e-6, "Distance is incorrect for f32.");
        assert!((grad[0] - -0.57735026).abs() < 1e-6, "Gradient[0] is incorrect for f32.");
        assert!((grad[1] - -0.57735026).abs() < 1e-6, "Gradient[1] is incorrect for f32.");
        assert!((grad[2] - -0.57735026).abs() < 1e-6, "Gradient[2] is incorrect for f32.");
    }

    #[test]
    fn test_euclidean_grad_zero_distance() {
        let x = vec![1.0f64, 2.0, 3.0];
        let y = vec![1.0f64, 2.0, 3.0];

        let (dist, grad) = euclidean_grad(&x, &y);
        assert!((dist - 0.0).abs() < 1e-6, "Distance should be 0 for identical vectors.");
        for &g in grad.iter() {
            assert!((g - 0.0).abs() < 1e-6, "Gradient should be 0 for identical vectors.");
        }
    }

    #[test]
    #[should_panic(expected = "Input arrays must have the same length.")]
    fn test_euclidean_grad_different_lengths() {
        let x = vec![1.0f64, 2.0];
        let y = vec![4.0f64, 5.0, 6.0];
        euclidean_grad(&x, &y); // This should panic
    }
}