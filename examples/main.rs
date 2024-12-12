use fast_distances::standardised_euclidean;
use ndarray::arr1;

fn main() {
    let x = arr1(&[1.0f32, 2.0f32, 3.0f32]);
    let y = arr1(&[4.0f32, 5.0f32, 6.0f32]);

    let distance = standardised_euclidean(&x.view(), &y.view(), None);

    println!("distance - {distance}");
}
