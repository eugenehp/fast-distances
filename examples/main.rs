use fast_distances::poincare;
use ndarray::arr1;

fn main() {
    // Example usage with f64
    let u_f64 = arr1(&[0.5, 0.3, 0.2]);
    let v_f64 = arr1(&[0.1, 0.4, 0.5]);
    let dist_f64 = poincare(&u_f64, &v_f64);
    println!("Poincare distance (f64): {}", dist_f64);

    // Example usage with f32
    let u_f32 = arr1(&[0.5f32, 0.3f32, 0.2f32]);
    let v_f32 = arr1(&[0.1f32, 0.4f32, 0.5f32]);
    let dist_f32 = poincare(&u_f32, &v_f32);
    println!("Poincare distance (f32): {}", dist_f32);
}
