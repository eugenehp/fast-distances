use ndarray::ArrayView1;

/// Computes the Hamming distance between two vectors `x` and `y`.
///
/// The Hamming distance is a measure of the number of positions at which the corresponding elements of two
/// sequences of equal length are different. This implementation assumes that `x` and `y` are binary vectors (0 or 1).
///
/// The formula for the Hamming distance is:
///
/// ..math::
///     D(x, y) = \frac{\text{number of different bits}}{\text{length of the vector}}
///
/// # Parameters:
/// - `x`: A reference to an `ArrayView1<T>` representing the first vector.
/// - `y`: A reference to an `ArrayView1<T>` representing the second vector.
///
/// # Returns:
/// The normalized Hamming distance between `x` and `y` as a scalar of type `T`.
///
/// # Panics:
/// - This function may panic if the lengths of `x` and `y` do not match.
///
/// # Example:
/// ```rust
/// use ndarray::arr1;
/// use fast_distances::*;
/// let x = arr1(&[1, 0, 1, 0]);
/// let y = arr1(&[0, 0, 1, 1]);
/// let dist = hamming(&x.view(), &y.view());
/// println!("Hamming Distance: {}", dist);
/// ```
pub fn hamming<T>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> f64
where
    T: PartialEq,
{
    // Ensure that the vectors x and y have the same length.
    assert_eq!(x.len(), y.len(), "Vectors must have the same length.");

    // Count the number of differing positions
    let mut result = 0.0;
    for i in 0..x.len() {
        if x[i] != y[i] {
            result += 1.0;
        }
    }

    // Return the normalized Hamming distance (number of differing positions divided by the length of the vector)
    result / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_hamming_basic() {
        // Test with binary vectors of equal length
        let x = arr1(&[1, 0, 1, 0]);
        let y = arr1(&[0, 0, 1, 1]);

        let dist = hamming(&x.view(), &y.view());
        let expected_dist = 2.0 / 4.0; // 2 differing positions out of 4 elements
        assert!((dist - expected_dist).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_identical_vectors() {
        // Test with identical vectors (Hamming distance should be 0)
        let x = arr1(&[1, 0, 1, 0]);
        let y = arr1(&[1, 0, 1, 0]);

        let dist = hamming(&x.view(), &y.view());
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_hamming_max_distance() {
        // Test with vectors that have the maximum possible distance (all elements different)
        let x = arr1(&[1, 1, 1, 1]);
        let y = arr1(&[0, 0, 0, 0]);

        let dist = hamming(&x.view(), &y.view());
        assert_eq!(dist, 1.0); // All positions are different
    }

    #[test]
    fn test_hamming_empty_vectors() {
        // Test with empty vectors (distance should be 0)
        let x = arr1::<i32>(&[]);
        let y = arr1::<i32>(&[]);

        let dist = hamming(&x.view(), &y.view());
        assert!(dist.is_nan()); // No elements, so distance is 0
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length.")]
    fn test_hamming_different_length_vectors() {
        // Test with vectors of different lengths (should panic)
        let x = arr1(&[1, 0, 1]);
        let y = arr1(&[0, 0]);

        hamming(&x.view(), &y.view());
    }
}
