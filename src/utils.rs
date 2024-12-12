use ndarray;

use ndarray::{Array1, Array2};
use num::{Float, Zero};

/// sign is a function that takes a floating-point number (f64 in this case) as input and returns an integer (i32).
/// If a is less than 0.0, it returns -1, otherwise it returns 1.
pub fn sign<T>(a: T) -> i32
where
    T: Zero + PartialOrd + Copy,
{
    if a < T::zero() {
        -1
    } else {
        1
    }
}

// Function to generate an identity matrix of size n x n for a given type T
pub fn identity_matrix<T>(n: usize) -> Array2<T>
where
    T: Float,
{
    let mut identity = Array2::<T>::zeros((n, n));
    for i in 0..n {
        identity[(i, i)] = T::one(); // Set diagonal elements to 1.0
    }
    identity
}

// Function to generate a ones vector of size n for a given type T
pub fn ones_vector<T>(n: usize) -> Array1<T>
where
    T: Float,
{
    Array1::<T>::ones(n)
}

// Function to generate a cost matrix (1.0 - identity matrix) of size n x n for a given type T
pub fn cost_matrix<T>(n: usize) -> Array2<T>
where
    T: Float,
{
    let identity = identity_matrix::<T>(n);
    Array2::<T>::from_elem((n, n), T::one()) - identity
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for sign with f64
    #[test]
    fn test_sign_f64() {
        assert_eq!(sign(-3.5), -1); // Negative number
        assert_eq!(sign(0.0), 1); // Zero
        assert_eq!(sign(4.5), 1); // Positive number
    }

    // Test for sign with f32
    #[test]
    fn test_sign_f32() {
        assert_eq!(sign(-2.7f32), -1); // Negative number
        assert_eq!(sign(0.0f32), 1); // Zero
        assert_eq!(sign(3.3f32), 1); // Positive number
    }

    // Test for sign with i32
    #[test]
    fn test_sign_i32() {
        assert_eq!(sign(-10), -1); // Negative number
        assert_eq!(sign(0), 1); // Zero
        assert_eq!(sign(25), 1); // Positive number
    }

    // Test for sign with edge cases
    #[test]
    fn test_sign_edge_cases() {
        // Test with large values
        assert_eq!(sign(f64::INFINITY), 1); // Positive infinity
        assert_eq!(sign(f64::NEG_INFINITY), -1); // Negative infinity
        assert_eq!(sign(f64::NAN), 1); // NaN (Not a Number), treated as non-negative
    }

    // Test identity matrix for f64
    #[test]
    fn test_identity_matrix_f64() {
        let n = 3;
        let identity = identity_matrix::<f64>(n);

        // Assert diagonal elements are 1.0 and others are 0.0
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(
                        identity[(i, j)],
                        1.0,
                        "Diagonal element at ({}, {}) should be 1.0",
                        i,
                        j
                    );
                } else {
                    assert_eq!(
                        identity[(i, j)],
                        0.0,
                        "Off-diagonal element at ({}, {}) should be 0.0",
                        i,
                        j
                    );
                }
            }
        }
    }

    // Test identity matrix for f32
    #[test]
    fn test_identity_matrix_f32() {
        let n = 2;
        let identity = identity_matrix::<f32>(n);

        // Assert diagonal elements are 1.0 and others are 0.0
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert_eq!(
                        identity[(i, j)],
                        1.0,
                        "Diagonal element at ({}, {}) should be 1.0",
                        i,
                        j
                    );
                } else {
                    assert_eq!(
                        identity[(i, j)],
                        0.0,
                        "Off-diagonal element at ({}, {}) should be 0.0",
                        i,
                        j
                    );
                }
            }
        }
    }

    // Test ones vector for f64
    #[test]
    fn test_ones_vector_f64() {
        let n = 4;
        let ones = ones_vector::<f64>(n);

        // Assert all elements are 1.0
        for i in 0..n {
            assert_eq!(ones[i], 1.0, "Element at index {} should be 1.0", i);
        }
    }

    // Test ones vector for f32
    #[test]
    fn test_ones_vector_f32() {
        let n = 3;
        let ones = ones_vector::<f32>(n);

        // Assert all elements are 1.0
        for i in 0..n {
            assert_eq!(ones[i], 1.0, "Element at index {} should be 1.0", i);
        }
    }

    // Test cost matrix for f64
    #[test]
    fn test_cost_matrix_f64() {
        let n = 3;
        let cost = cost_matrix::<f64>(n);
        let identity = identity_matrix::<f64>(n);

        // Assert that cost matrix is 1.0 - identity matrix
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    cost[(i, j)],
                    1.0 - identity[(i, j)],
                    "Element at ({}, {}) should be 1.0 - identity",
                    i,
                    j
                );
            }
        }
    }

    // Test cost matrix for f32
    #[test]
    fn test_cost_matrix_f32() {
        let n = 2;
        let cost = cost_matrix::<f32>(n);
        let identity = identity_matrix::<f32>(n);

        // Assert that cost matrix is 1.0 - identity matrix
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    cost[(i, j)],
                    1.0 - identity[(i, j)],
                    "Element at ({}, {}) should be 1.0 - identity",
                    i,
                    j
                );
            }
        }
    }

    // Test edge case: identity matrix for 1x1 (f64)
    #[test]
    fn test_identity_matrix_1x1_f64() {
        let n = 1;
        let identity = identity_matrix::<f64>(n);

        // Assert that it is a 1x1 matrix with 1.0
        assert_eq!(identity[(0, 0)], 1.0);
    }

    // Test edge case: identity matrix for 1x1 (f32)
    #[test]
    fn test_identity_matrix_1x1_f32() {
        let n = 1;
        let identity = identity_matrix::<f32>(n);

        // Assert that it is a 1x1 matrix with 1.0
        assert_eq!(identity[(0, 0)], 1.0);
    }
}
