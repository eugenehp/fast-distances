use fast_distances::manhattan;
use ndarray::arr1;

fn main() {
    let x = arr1(&[1.0, 2.0, 3.0]);
    let y = arr1(&[4.0, 5.0, 6.0]);
    println!("{}", manhattan(&x.view(), &y.view()));

    let a = arr1(&[-1.0, 0.0, 2.0]);
    let b = arr1(&[3.0, -4.0, 5.0]);
    println!("{}", manhattan(&a.view(), &b.view()));

    let u = arr1(&[0.0; 3]);
    let v = arr1(&[0.0; 3]);
    println!("{}", manhattan(&u.view(), &v.view()));
}
