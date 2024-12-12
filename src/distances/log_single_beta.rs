use std::f64::consts::PI;

use num::Float;

/// Approximate the log of the single Beta function, as defined in the given Python function.
pub fn log_single_beta<T: Float>(x: T) -> T {
    T::ln(T::from(2.0).unwrap()) * (-T::from(2.0).unwrap() * x + T::from(0.5).unwrap())
        + T::from(0.5).unwrap() * (T::from(2.0).unwrap() * T::from(PI).unwrap() / x).ln()
        + T::from(0.125).unwrap() / x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_single_beta_f32() {
        // Test for x = 1.0
        let result = log_single_beta(1.0f32);
        assert_eq!(result, 0.004217744, "log_single_beta(1.0) ≈ 0.004217744");

        // Test for x = 2.0
        let result = log_single_beta(2.0f32);
        assert_eq!(result, -1.7911501, "log_single_beta(2.0) ≈ -1.7911501");

        // Test for x = 3.0
        let result = log_single_beta(3.0f32);
        assert_eq!(result, -3.4010103, "log_single_beta(3.0) ≈ -3.4010103");
    }

    #[test]
    fn test_log_single_beta_f64() {
        // Test for x = 1.0
        let result = log_single_beta(1.0f64);
        assert_eq!(
            result, 0.004217762364754796,
            "log_single_beta(1.0) ≈ 0.004217762364754796"
        );

        // Test for x = 2.0
        let result = log_single_beta(2.0f64);
        assert_eq!(
            result, -1.7911501890351085,
            "log_single_beta(2.0) ≈ -1.7911501890351085"
        );

        // Test for x = 3.0
        let result = log_single_beta(3.0f64);
        assert_eq!(
            result, -3.401010437542415,
            "log_single_beta(3.0) ≈ -3.401010437542415"
        );
    }
}
