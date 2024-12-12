use ndarray::ArrayView1;

/// Computes the Canberra distance between two vectors `x` and `y`.
///
/// The Canberra distance is defined as:
///
/// ..math::
///     D(x, y) = \sum_i \frac{|x_i - y_i|}{|x_i| + |y_i|}
///
/// where `x_i` and `y_i` are the elements of the vectors `x` and `y` respectively, and the sum runs over all dimensions.
///
/// # Parameters:
/// - `x`: A reference to an `ArrayView1<T>` representing the first vector.
/// - `y`: A reference to an `ArrayView1<T>` representing the second vector.
///
/// # Returns:
/// The Canberra distance between `x` and `y` as a scalar of type `T`.
///
/// # Panics:
/// - This function may panic if the lengths of `x` and `y` do not match.
///
/// # Example:
/// ```rust
/// use ndarray::arr1;
/// use fast_distances::*;
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// let dist = canberra(&x.view(), &y.view());
/// println!("Canberra Distance: {}", dist);
/// ```
pub fn canberra<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: num::Float,
{
    // Ensure that the vectors x and y have the same length.
    assert_eq!(x.len(), y.len(), "Vectors must have the same length.");

    // Initialize result
    let mut result: T = T::zero();

    // Loop through the elements of the vectors
    for i in 0..x.len() {
        let denominator = x[i].abs() + y[i].abs();
        if denominator > T::zero() {
            result = result + (x[i] - y[i]).abs() / denominator;
        }
    }

    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_canberra_basic() {
        // Test with simple vectors
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);

        let dist = canberra(&x.view(), &y.view());
        let expected_dist = (3.0 / 5.0) + (3.0 / 7.0) + (3.0 / 9.0); // (|1-4|/|1+4|) + (|2-5|/|2+5|) + (|3-6|/|3+6|)
        assert_eq!(dist, expected_dist);
    }

    #[test]
    fn test_canberra_identical_vectors() {
        // Test with identical vectors (Canberra distance should be 0)
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let dist = canberra(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_canberra_zero_elements() {
        // Test with vectors that have zero elements (denominator should avoid division by zero)
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = arr1(&[1.0, 2.0, 3.0]);

        let dist = canberra(&x.view(), &y.view());
        let expected_dist = 1.0 + 1.0 + 1.0; // As numerator is non-zero and denominator has non-zero value
        assert_eq!(dist, expected_dist);
    }

    #[test]
    fn test_canberra_empty_vectors() {
        // Test with empty vectors (distance should be 0)
        let x = arr1::<f64>(&[]);
        let y = arr1::<f64>(&[]);

        let dist = canberra(&x.view(), &y.view());
        assert_eq!(dist, 0.0); // No elements, so distance is 0
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length.")]
    fn test_canberra_different_length_vectors() {
        // Test with vectors of different lengths (should panic)
        let x = arr1(&[1.0, 2.0]);
        let y = arr1(&[0.0, 0.0, 0.0]);

        canberra(&x.view(), &y.view());
    }
}
