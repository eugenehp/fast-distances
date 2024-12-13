# fast-distances

Rust Similarity and Distance Metrics Library

This Rust package provides a wide range of functions for computing various distance and similarity metrics between vectors or points in a high-dimensional space. These metrics are widely used in fields such as machine learning, statistics, data science, and computational biology.

## Modules

Each module in this package implements a specific distance or similarity measure, some with gradient computations for optimization tasks. Below is a list of available modules:

* approx_log_gamma: Approximation of the logarithm of the Gamma function.
* bray_curtis: Bray-Curtis dissimilarity, a measure for ecological distance.
* bray_curtis_grad: Gradient of the Bray-Curtis dissimilarity.
* canberra: Canberra distance, a city block-like metric with a normalization.
* canberra_grad: Gradient of the Canberra distance.
* chebyshev: Chebyshev distance (L∞ distance), the maximum distance along any coordinate axis.
* chebyshev_grad: Gradient of the Chebyshev distance.
* correlation: Pearson correlation coefficient, a measure of linear correlation between two vectors.
* cosine: Cosine similarity, measuring the cosine of the angle between two vectors.
* cosine_grad: Gradient of the cosine similarity.
* dice: Dice coefficient, a similarity measure often used in bioinformatics.
* euclidean: Euclidean distance, the straight-line distance between two points.
* euclidean_grad: Gradient of the Euclidean distance.
* hamming: Hamming distance, the number of differing positions between two strings of equal length.
* haversine: Haversine distance, used to calculate the great-circle distance between two points on a sphere.
* haversine_grad: Gradient of the Haversine distance.
* hellinger: Hellinger distance, a measure for comparing probability distributions.
* hellinger_grad: Gradient of the Hellinger distance.
* hyperboloid_grad: Gradient of the hyperboloid distance, a metric on hyperbolic spaces.
* jaccard: Jaccard similarity coefficient, a measure of the intersection between two sets divided by their union.
* kulsinski: Kulsinski similarity coefficient, a distance measure for binary vectors.
* ll_dirichlet: Log-Likelihood of the Dirichlet distribution, used for probabilistic comparison of Dirichlet-distributed data.
* log_beta: Log of the Beta distribution, used in statistical modeling.
* log_single_beta: Logarithmic computation of a single Beta distribution.
* mahalanobis: Mahalanobis distance, a distance metric that accounts for correlations between variables.
* mahalanobis_grad: Gradient of the Mahalanobis distance.
* manhattan: Manhattan distance (L1 distance), the sum of the absolute differences between coordinates.
* manhattan_grad: Gradient of the Manhattan distance.
* matching: Matching distance, a similarity measure based on matching elements in two sets.
* minkowski: Minkowski distance, a generalization of both Euclidean and Manhattan distances.
* minkowski_grad: Gradient of the Minkowski distance.
* poincare: Poincaré distance, used for hyperbolic spaces and geometries.
* rogers_tanimoto: Rogers-Tanimoto similarity, a distance measure for binary data.
* russellrao: Russell-Rao similarity, a measure for binary vectors.
* sokal_michener: Sokal-Michener similarity, a metric for categorical data.
* sokal_sneath: Sokal-Sneath similarity, another metric for categorical data.
* standardised_euclidean: Standardized Euclidean distance, which normalizes the Euclidean distance by the variance.
* standardised_euclidean_grad: Gradient of the standardized Euclidean distance.
* weighted_minkowski: Weighted Minkowski distance, a variant of Minkowski with weightings for each dimension.
* weighted_minkowski_grad: Gradient of the weighted Minkowski distance.
* yule: Yule's coefficient, used to measure association between two binary vectors.


## Installation

Add this package to your Cargo.toml to use it in your project:

```toml
[dependencies]
fast-distances = "0.1"
```

## Usage

To use one of the available distance or similarity metrics, import the respective module in your Rust code:

```rust
use distances::{cosine, euclidean, manhattan};

fn main() {
    let vector1 = vec![1.0, 2.0, 3.0];
    let vector2 = vec![4.0, 5.0, 6.0];

    // Compute cosine similarity
    let cosine_sim = cosine(&vector1, &vector2);
    println!("Cosine Similarity: {}", cosine_sim);

    // Compute Euclidean distance
    let euclidean_dist = euclidean(&vector1, &vector2);
    println!("Euclidean Distance: {}", euclidean_dist);

    // Compute Manhattan distance
    let manhattan_dist = manhattan(&vector1, &vector2);
    println!("Manhattan Distance: {}", manhattan_dist);
}
```

## Contributing

Contributions are welcome! If you'd like to contribute a new metric or improve an existing one, feel free to open an issue or a pull request.

1. Fork the repository.
2. Clone your fork locally.
3. Make changes and run tests to ensure they pass.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This package draws from many well-established distance and similarity metrics commonly used in data analysis, machine learning, and information retrieval.