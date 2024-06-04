use mscore::algorithm::utility::adaptive_integration;

fn main() {
    let f = |x: f64| 1.0 / (x * x + 1.0); // Example: Integrating the function 1/(x^2 + 1)
    let a = 0.0;
    let b = 1.0;
    let epsabs = 1e-4;
    let epsrel = 1e-4;

    let (result, total_error) = adaptive_integration(&f, a, b, epsabs, epsrel);
    println!("Result: {}", result);
    println!("Total Estimated Error: {}", total_error);
}
