extern crate ndarray;

use ndarray::ArrayView1;
use num::Float;

/// Computes the Haversine distance between two points on the Earth's surface.
///
/// The Haversine distance is defined as:
///
/// ..math::
///     D(x, y) = 2 \times \arcsin\left(\sqrt{\sin^2\left(\frac{x_1 - y_1}{2}\right) + \cos(x_1) \times \cos(y_1) \times \sin^2\left(\frac{x_2 - y_2}{2}\right)}\right)
///
/// Where:
/// - `x` and `y` are two points in 2D space, represented as latitude and longitude.
///
/// # Arguments
/// * `x` - A 1D array (view) of values representing the first point (latitude, longitude).
/// * `y` - A 1D array (view) of values representing the second point (latitude, longitude).
///
/// # Returns
/// A f64 value representing the Haversine distance between `x` and `y` in radians.
pub fn haversine<T: Float>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T {
    if x.len() != 2 || y.len() != 2 {
        panic!("Haversine is only defined for 2-dimensional data");
    }

    let sin_lat = (T::from(0.5).unwrap() * (x[0] - y[0])).sin();
    let sin_long = (T::from(0.5).unwrap() * (x[1] - y[1])).sin();
    let result = (sin_lat.powi(2) + x[0].cos() * y[0].cos() * sin_long.powi(2)).sqrt();
    T::from(2.0).unwrap() * result.asin()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_haversine_basic() {
        // Test for two points on Earth (latitude and longitude in radians)
        let x = arr1(&[0.0, 0.0]); // Point at the origin (0, 0)
        let y = arr1(&[0.0, PI / 2.0]); // Point at (0, 90 degrees)

        let distance = haversine(&x.view(), &y.view());
        let expected_distance = 1.5707963267948963;
        assert_eq!(distance, expected_distance);
    }

    #[test]
    fn test_haversine_identical_points() {
        // Two identical points (should return distance = 0)
        let x = arr1(&[0.0, 0.0]);
        let y = arr1(&[0.0, 0.0]);

        let distance = haversine(&x.view(), &y.view());
        assert_eq!(distance, 0.0); // Same points should result in 0 distance
    }

    #[test]
    fn test_haversine_opposite_points() {
        // Two opposite points (should return maximum distance on a great circle)
        let x = arr1(&[PI / 2.0, 0.0]); // (90, 0)
        let y = arr1(&[-PI / 2.0, 0.0]); // (-90, 0)

        let distance = haversine(&x.view(), &y.view());
        let expected_distance = PI; // Half of the Earth's circumference
        assert_eq!(distance, expected_distance);
    }

    #[test]
    fn test_haversine_different_longitudes() {
        // Two points with the same latitude, different longitudes
        let x = arr1(&[0.0, 0.0]); // (0, 0)
        let y = arr1(&[0.0, PI]); // (0, 180)

        let distance = haversine(&x.view(), &y.view());
        let expected_distance = PI; // Points on the same latitude, but different longitudes
        assert_eq!(distance, expected_distance);
    }

    #[test]
    fn test_haversine_invalid_dimension() {
        // Test for invalid input dimensions (e.g., 3D data)
        let x = arr1(&[1.0, 2.0, 3.0]); // Invalid input (3D)
        let y = arr1(&[1.0, 2.0, 3.0]);

        // The function should panic with an error message
        let result = std::panic::catch_unwind(|| {
            haversine(&x.view(), &y.view());
        });

        assert!(result.is_err());
    }
}
