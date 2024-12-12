use ndarray::ArrayView1;
use num::Float;

/// Computes the Manhattan, taxicab, or L1 distance between two vectors.
///
/// # Arguments
///
/// * `x` - A one-dimensional array view of type `T`.
/// * `y` - A one-dimensional array view of type `T`.
///
/// # Returns
///
/// The Manhattan distance as a value of type `T`.
///
/// # Example
///
/// ```
/// use ndarray::arr1;
/// use fast_distances::manhattan;
///
/// let x = arr1(&[1.0, 2.0, 3.0]);
/// let y = arr1(&[4.0, 5.0, 6.0]);
/// assert_eq!(manhattan(&x.view(), &y.view()), 9.0);
/// ```
///
/// # Mathematical Definition
///
/// ..math::
///     D(x, y) = \sum_i |x_i - y_i|
pub fn manhattan<T: Float>(x: &ArrayView1<T>, y: &ArrayView1<T>) -> T
where
    T: Float + std::iter::Sum,
{
    assert_eq!(x.len(), y.len(), "Input vectors must have the same length");

    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).abs())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_manhattan() {
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = arr1(&[4.0, 5.0, 6.0]);
        assert_eq!(manhattan(&x.view(), &y.view()), 9.0);

        let a = arr1(&[-1.0, 0.0, 2.0]);
        let b = arr1(&[3.0, -4.0, 5.0]);
        assert_eq!(manhattan(&a.view(), &b.view()), 11.0);

        let u = arr1(&[0.0; 3]);
        let v = arr1(&[0.0; 3]);
        assert_eq!(manhattan(&u.view(), &v.view()), 0.0);
    }

    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_manhattan_different_lengths() {
        let x = arr1(&[1.0, 2.0]);
        let y = arr1(&[3.0, 4.0, 5.0]);
        manhattan(&x.view(), &y.view());
    }
}
