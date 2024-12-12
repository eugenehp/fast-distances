use std::{f64::consts::PI, iter::Sum};

use num::Float;

fn log_single_beta<T: Float>(x: T) -> T {
    T::ln(T::from(2.0).unwrap()) * (-T::from(2.0).unwrap() * x + T::from(0.5).unwrap())
        + T::from(0.5).unwrap() * (T::from(2.0).unwrap() * T::from(PI).unwrap() / x).ln()
        + T::from(0.125).unwrap() / x
}

fn log_beta<T: Float>(x: T, y: T) -> T
where
    T: Float,
{
    let a = x.min(y);
    let b = x.max(y);

    if b < T::from(5.0).unwrap() {
        let mut value = -T::ln(b);
        for i in 1..a.to_i64().unwrap() {
            let ii = T::from(i).unwrap();
            value = value + T::ln(ii) - T::ln(b + ii);
        }
        value
    } else {
        log_single_beta(x) + log_single_beta(y) - log_single_beta(x + y)
    }
}

/// Calculates the symmetric relative log likelihood (log Dirichlet likelihood) of rolling
/// `data2` versus `data1` in `n2` trials on a die that rolled `data1` in `n1` trials.
///
/// The formula used is based on the Dirichlet-Multinomial model, and it computes the difference
/// in likelihood between the two sets of data under a Dirichlet distribution assumption. This
/// measure is useful for comparing the distribution of counts between two categorical datasets,
/// typically for hypothesis testing or evaluating model performance when categorical data is involved.
///
/// The equation is as follows:
///
/// ..math::
///     D(data1, data2) = \sqrt{ \frac{1}{n2} \left( \log \beta(data1, data2) - \log \beta(n1, n2) - ( \text{self\_denom2} - \log \text{single\_beta}(n2) ) \right) + \frac{1}{n1} \left( \log \beta(data2, data1) - \log \beta(n2, n1) - ( \text{self\_denom1} - \log \text{single\_beta}(n1) ) \right) }
///
/// # Arguments
///
/// * `data1` - A slice of `T` values representing the first data set (e.g., counts from one die roll).
/// * `data2` - A slice of `T` values representing the second data set (e.g., counts from another die roll).
///
/// # Returns
///
/// Returns a `T` value representing the log likelihood of `data2` relative to `data1`. A higher value indicates that `data2` is more likely given `data1`.
///
/// # Examples
///
/// ```
/// use fast_distances::*;
/// let data1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
/// let data2: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
/// let result = ll_dirichlet(&data1, &data2);
/// println!("Log Dirichlet likelihood: {}", result);
/// ```
pub fn ll_dirichlet<T: Float>(data1: &[T], data2: &[T]) -> T
where
    T: Float + Sum,
{
    let n1: T = data1.iter().copied().sum();
    let n2: T = data2.iter().copied().sum();

    let mut log_b = T::from(0.0).unwrap();
    let mut self_denom1 = T::from(0.0).unwrap();
    let mut self_denom2 = T::from(0.0).unwrap();

    for i in 0..data1.len() {
        if data1[i] * data2[i] > T::from(0.9).unwrap() {
            log_b = log_b + log_beta(data1[i], data2[i]);
            self_denom1 = self_denom1 + log_single_beta(data1[i]);
            self_denom2 = self_denom2 + log_single_beta(data2[i]);
        } else {
            if data1[i] > T::from(0.9).unwrap() {
                self_denom1 = self_denom1 + log_single_beta(data1[i]);
            }

            if data2[i] > T::from(0.9).unwrap() {
                self_denom2 = self_denom2 + log_single_beta(data2[i]);
            }
        }
    }

    T::sqrt(
        T::from(1.0).unwrap() / n2
            * (log_b - log_beta(n1, n2) - (self_denom2 - log_single_beta(n2)))
            + T::from(1.0).unwrap() / n1
                * (log_b - log_beta(n2, n1) - (self_denom1 - log_single_beta(n1))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ll_dirichlet_f32() {
        let data1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data2: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

        let result = ll_dirichlet(&data1, &data2);
        assert_eq!(result, 0.36789307, "ll_dirichlet with f32");
    }

    #[test]
    fn test_ll_dirichlet_f64() {
        let data1: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let data2: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];

        let result = ll_dirichlet(&data1, &data2);
        assert_eq!(result, 0.36789301898248805, "ll_dirichlet with f64");
    }
}
