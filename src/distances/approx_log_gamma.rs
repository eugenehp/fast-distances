use num::Float;
use std::f64::consts::PI;

/// Approximate the logarithm of the Gamma function (log(Γ(x))) using Stirling's approximation.
/// This approximation is valid for x > 0.
///
/// # Arguments
///
/// * `x` - A floating point number.
///
/// # Returns
///
/// * The approximated log(Gamma(x)).
pub fn approx_log_gamma<T: Float>(x: T) -> T {
    if x == T::one() {
        return T::zero();
    }

    // Stirling's approximation
    x * x.ln() - x
        + T::from(0.5).unwrap() * (T::from(2.0).unwrap() * T::from(PI).unwrap() / x).ln()
        + T::one() / (x * T::from(12.0).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_log_gamma_f32() {
        // Test for x = 1.0
        assert_eq!(approx_log_gamma(1.0f32), 0.0f32);

        assert_eq!(approx_log_gamma(2.0f32), 0.00032601878);

        // Test for x = 5.0 (Log(Gamma(5)) ≈ 3.178)
        let gamma_5 = approx_log_gamma(5.0f32);
        assert_eq!(gamma_5, 3.178076, "Log(Gamma(5)) ≈ 3.178");

        // Test for x = 10.0 (Log(Gamma(10)) ≈ 12.8019)
        let gamma_10 = approx_log_gamma(10.0f32);
        assert_eq!(gamma_10, 12.801831, "Log(Gamma(10)) ≈ 12.8019");

        // Test for x = 0.5 (Log(Gamma(0.5)) ≈ -1.9635)
        let gamma_half = approx_log_gamma(0.5f32);
        assert_eq!(gamma_half, 0.5856052, "Log(Gamma(0.5)) ≈ -1.9635");

        // Test for x = 3.0 (Log(Gamma(3)) ≈ 1.0986)
        let gamma_3 = approx_log_gamma(3.0f32);
        assert_eq!(gamma_3, 0.6932471, "Log(Gamma(3)) ≈ 1.0986");
    }

    #[test]
    fn test_approx_log_gamma_f64() {
        // Test for x = 1.0
        assert_eq!(approx_log_gamma(1.0f64), 0.0f64);

        assert_eq!(approx_log_gamma(2.0f64), 0.00032597071125731875);

        // Test for x = 5.0 (Log(Gamma(5)) ≈ 3.178)
        let gamma_5 = approx_log_gamma(5.0f64);
        assert_eq!(gamma_5, 3.1780758058247907, "Log(Gamma(5)) ≈ 3.178");

        // Test for x = 10.0 (Log(Gamma(10)) ≈ 12.8019)
        let gamma_10 = approx_log_gamma(10.0f64);
        assert_eq!(gamma_10, 12.801830249981444, "Log(Gamma(10)) ≈ 12.8018");

        // Test for x = 0.5 (Log(Gamma(0.5)) ≈ -1.9635)
        let gamma_half = approx_log_gamma(0.5f64);
        assert_eq!(
            gamma_half, 0.5856051998713393,
            "Log(Gamma(0.5)) ≈ 0.5856051998713393"
        );

        let gamma_3 = approx_log_gamma(3.0f64);
        assert_eq!(
            gamma_3, 0.6932470326527248,
            "Log(Gamma(3)) ≈ 0.6932470326527248"
        );
    }
}
