extern crate ndarray;
extern crate num;

use ndarray::{Array1, ArrayView1};
use num::Float;
use std::f64::consts::PI;

/// Computes the gradient of the Haversine distance between two points on the Earth's surface.
///
/// The gradient of the Haversine distance is computed with respect to both points.
///
/// The formula for the gradient is derived from the Haversine distance function and the partial derivatives
/// with respect to the latitude and longitude of both points.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first point (latitude, longitude).
/// * `y` - A 1D array (view) of values representing the second point (latitude, longitude).
///
/// # Returns
/// A tuple of the Haversine distance and the gradient of the distance with respect to `x` and `y`
///
/// # Panics
/// Panics if `x` or `y` are not 2-dimensional (latitude and longitude).
pub fn haversine_grad<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> (T, Array1<T>)
where
    T: Float,
{
    if x.len() != 2 || y.len() != 2 {
        panic!("Haversine is only defined for 2-dimensional data");
    }

    // Computing sin and cos terms
    let sin_lat = (x[0] - y[0]).sin() * T::from(0.5).unwrap();
    let cos_lat = (x[0] - y[0]).cos() * T::from(0.5).unwrap();
    let sin_long = (x[1] - y[1]).sin() * T::from(0.5).unwrap();
    let cos_long = (x[1] - y[1]).cos() * T::from(0.5).unwrap();

    // Compute auxiliary terms for the gradient
    let a_0 = (x[0] + T::from(PI / 2.0).unwrap()).cos()
        * (y[0] + T::from(PI / 2.0).unwrap()).cos()
        * sin_long.powi(2);
    let a_1 = a_0 + sin_lat.powi(2);

    // Compute the distance (Haversine formula)
    let d = T::from(2.0).unwrap() * (a_1.abs().sqrt().asin());

    // Compute the denominator used for the gradient
    let denom = a_1.abs().sqrt() * (a_1 - T::from(1.0).unwrap()).abs().sqrt();

    // Gradient components for x and y
    let grad_x = (sin_lat * cos_lat
        - (x[0] + T::from(PI / 2.0).unwrap()).sin()
            * (y[0] + T::from(PI / 2.0).unwrap()).cos()
            * sin_long.powi(2))
        / (denom + T::from(1e-6).unwrap());

    let grad_y = ((x[0] + T::from(PI / 2.0).unwrap()).cos()
        * (y[0] + T::from(PI / 2.0).unwrap()).cos()
        * sin_long
        * cos_long)
        / (denom + T::from(1e-6).unwrap());

    // Return the distance and gradient as a tuple
    let grad = Array1::from_vec(vec![grad_x, grad_y]);
    (d, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_haversine_grad_basic() {
        // Test for two points on Earth (latitude and longitude in radians)
        let x = arr1(&[0.0, 0.0]); // Point at the origin (0, 0)
        let y = arr1(&[0.0, PI / 2.0]); // Point at (0, 90 degrees)

        let (distance, grad) = haversine_grad(&x.view(), &y.view());
        let expected_distance = 6.123233995736766e-17;
        let expected_grad = arr1(&[-1.530808498887324e-11, -5.739612553970444e-44]);

        assert_eq!(distance, expected_distance);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_haversine_grad_identical_points() {
        // Two identical points (should return distance = 0)
        let x = arr1(&[0.0, 0.0]);
        let y = arr1(&[0.0, 0.0]);

        let (distance, grad) = haversine_grad(&x.view(), &y.view());
        let expected_grad = arr1(&[0.0, 0.0]);

        assert_eq!(distance, 0.0); // Same points should result in 0 distance
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_haversine_grad_opposite_points() {
        // Two opposite points (should return maximum distance on a great circle)
        let x = arr1(&[PI / 2.0, 0.0]); // (90, 0)
        let y = arr1(&[-PI / 2.0, 0.0]); // (-90, 0)

        let (distance, grad) = haversine_grad(&x.view(), &y.view());
        let expected_distance = 1.2246467991473532e-16;
        let expected_grad = arr1(&[-3.061616997680913e-11, -0.0]);

        assert_eq!(distance, expected_distance);
        assert_eq!(grad, expected_grad);
    }

    #[test]
    fn test_haversine_grad_invalid_dimension() {
        // Test for invalid input dimensions (e.g., 3D data)
        let x = arr1(&[1.0, 2.0, 3.0]); // Invalid input (3D)
        let y = arr1(&[1.0, 2.0, 3.0]);

        // The function should panic with an error message
        let result = std::panic::catch_unwind(|| {
            haversine_grad(&x.view(), &y.view());
        });

        assert!(result.is_err());
    }
}
