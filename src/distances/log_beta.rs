use std::f64::consts::PI;

use num::Float;

fn approx_log_gamma<T: Float>(x: T) -> T {
    if x == T::one() {
        return T::zero();
    }

    x * x.ln() - x
        + T::from(0.5).unwrap() * (T::from(2.0).unwrap() * T::from(PI).unwrap() / x).ln()
        + T::one() / (x * T::from(12.0).unwrap())
}

/// Approximate the logarithm of the Beta function (log(B(x, y))) using two cases:
/// - For small values of `b` (less than 5), we compute using the series expansion.
pub fn log_beta<T: Float>(x: T, y: T) -> T {
    let a = x.min(y);
    let b = x.max(y);

    if b < T::from(5.0).unwrap() {
        let mut value = -b.ln();
        let mut i = T::one();
        while i < a {
            value = value + (i.ln() - (b + i).ln());
            i = i + T::one();
        }
        return value;
    } else {
        approx_log_gamma(x) + approx_log_gamma(y) - approx_log_gamma(x + y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_beta_f32() {
        // Test for x = 1.0 and y = 2.0 (Beta(1, 2))
        let result = log_beta(1.0f32, 2.0f32);
        assert_eq!(result, -0.6931472f32, "log(Beta(1, 2)) ≈ -0.6931472");

        // Test for x = 3.0 and y = 5.0 (Beta(3, 5))
        let result = log_beta(3.0f32, 5.0f32);
        assert_eq!(result, -4.6538444, "log(Beta(3, 5)) ≈ -4.6538444");

        // Test for x = 4.0 and y = 6.0 (Beta(4, 6))
        let result = log_beta(4.0f32, 6.0f32);
        assert_eq!(result, -6.2225246, "log(Beta(4, 6)) ≈ -6.2225246");
    }

    #[test]
    fn test_log_beta_f64() {
        // Test for x = 1.0 and y = 2.0 (Beta(1, 2))
        let result = log_beta(1.0f64, 2.0f64);
        assert_eq!(result, -0.6931471805599453, "log(Beta(1, 2)) ≈ -0.6931472");

        // Test for x = 3.0 and y = 5.0 (Beta(3, 5))
        let result = log_beta(3.0f64, 5.0f64);
        assert_eq!(
            result, -4.653843923992591,
            "log(Beta(3, 5)) ≈ -4.653843923992591"
        );

        // Test for x = 4.0 and y = 6.0 (Beta(4, 6))
        let result = log_beta(4.0f64, 6.0f64);
        assert_eq!(
            result, -6.222523616675956,
            "log(Beta(4, 6)) ≈ -6.222523616675956"
        );
    }
}
