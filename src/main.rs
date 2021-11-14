use image::{png::PNGEncoder, ColorType};
use num::Complex;
use num_cpus;
use std::{env, fs::File, str::FromStr};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 5 {
        eprintln!("Usage: {} FILE PIXELS UPPERLEFT LOWERRIGHT", args[0]);
        eprintln!("Example: {} mendel.png 1000x750 -1.2,0.35 -1,0.2", args[0]);
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x').expect("Error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("Error parsing upper left corner point");
    let lower_right = parse_complex(&args[4]).expect("Error parsing lower right corner point");

    let mut pixels = vec![0; bounds.0 * bounds.1];

    let threads = num_cpus::get();
    let rows_per_band = bounds.1 / threads + 1;
    {
        let bands = pixels.chunks_mut(rows_per_band * bounds.0);
        crossbeam::scope(|spawner| {
            for (i, band) in bands.enumerate() {
                let top = rows_per_band * i;
                let height = band.len() / bounds.0;
                let band_bounds = (bounds.0, height);
                let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
                let band_lower_right =
                    pixel_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);

                spawner.spawn(move |_| {
                    render(band, band_bounds, band_upper_left, band_lower_right);
                });
            }
        })
        .unwrap();
    }

    write_image(&args[1], &pixels, bounds).expect("Error writing PNG file");
}

/// Determine if `c` is in the Mandelbrot set, using at most `limit` iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of iterations it took for `c`
/// to leave the circle of radius 2 centred at the origin. If we reach the iteration limit without
/// being able to demonstrate that `c` is not a member then return `None`.
fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };
    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c;
    }
    None
}

/// Specifies the type of error which occured upon attempting to parse a pair from a separated
/// string.
#[derive(Debug, Clone)]
enum PairParseErrorType {
    SeparatorNotFound,
    ValueParseError,
}

/// Indicates an error occured in parsing a pair of values from a separated string.
#[derive(Debug, Clone)]
struct PairParseError(PairParseErrorType);

/// Parse a pair of values from a string given a separator character.
///
/// For example, `(400, 600)` from `"400x600"` with separator `'x'`.
fn parse_pair<T: FromStr>(s: &str, separator: char) -> Result<(T, T), PairParseError> {
    match s.find(separator) {
        None => Err(PairParseError(PairParseErrorType::SeparatorNotFound)),
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Ok((l, r)),
            _ => Err(PairParseError(PairParseErrorType::ValueParseError)),
        },
    }
}

/// Parse a complex number from a comma separated pair of numbers in a string.
fn parse_complex<T: FromStr>(s: &str) -> Result<Complex<T>, PairParseError> {
    let (re, im) = parse_pair(s, ',')?;
    Ok(Complex { re, im })
}

/// Given the coordinates of a pixel, return the corresponding point in the complex plane.
///
/// `bounds` is a pair giving the width and height of the image in pixels.
/// `pixel` is a (column, row) pair indicating a pixel in the image.
/// `upper_left` and `lower_right` are points on the complex plane designating the area our image
/// covers.
fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );

    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64,
    }
}

/// Render a rectangle of the Mandelbrot set into a buffer of pixels.
///
/// `pixels` holds one grayscale pixel per byte.
/// `bounds` gives the width and height of the buffer.
/// `upper_left` and `lower_right` specify the points on the complex plane corresponding to their
/// respective corners of the pixel buffer.
fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            pixels[row * bounds.0 + column] = match escape_time(point, 255) {
                None => 0,
                Some(count) => 255 - count as u8,
            };
        }
    }
}

/// Write the buffer `pixels`, whose dimensions are given by `bounds`, to the file named `filename`
/// as a PNG.
fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;

    let encoder = PNGEncoder::new(output);
    encoder.encode(
        &pixels,
        bounds.0 as u32,
        bounds.1 as u32,
        ColorType::Gray(8),
    )?;
    Ok(())
}

#[cfg(test)]
mod parse_pair_tests {
    use super::*;

    #[test]
    fn err_on_empty_string() {
        match parse_pair::<i32>("", ',') {
            Ok(t) => panic!("Should have thrown an error but got Ok({:?})", t),
            Err(PairParseError(PairParseErrorType::SeparatorNotFound)) => {}
            other => panic!("Wrong error thrown. Got {:?}", other),
        }
    }

    #[test]
    fn err_on_right_missing() {
        match parse_pair::<i32>("10,", ',') {
            Ok(t) => panic!("Should have thrown an error but got Ok({:?})", t),
            Err(PairParseError(PairParseErrorType::ValueParseError)) => {}
            other => panic!("Wrong error thrown. Got {:?}", other),
        }
    }

    #[test]
    fn err_on_left_missing() {
        match parse_pair::<i32>(",10", ',') {
            Ok(t) => panic!("Should have thrown an error but got Ok({:?})", t),
            Err(PairParseError(PairParseErrorType::ValueParseError)) => {}
            other => panic!("Wrong error thrown. Got {:?}", other),
        }
    }

    #[test]
    fn returns_pair() {
        match parse_pair::<i32>("-10,-20", ',') {
            Ok(t) => assert_eq!(t, (-10, -20)),
            other => panic!("Error thrown. Got {:?}", other),
        }
    }

    #[test]
    fn returns_float_pair() {
        match parse_pair::<f32>("0.5,-2.25", ',') {
            Ok(t) => assert_eq!(t, (0.5, -2.25)),
            other => panic!("Error thrown. Got {:?}", other),
        }
    }
}

#[cfg(test)]
mod parse_complex_tests {
    use super::*;

    #[test]
    fn parses_complex() {
        match parse_complex::<f32>("-0.123,0.543") {
            Ok(complex) => {
                assert_eq!(
                    complex,
                    Complex {
                        re: -0.123,
                        im: 0.543
                    }
                )
            }
            other => panic!("Error thrown. Got {:?}", other),
        }
    }

    #[test]
    fn propagates_errors() {
        match parse_complex::<f32>(",0.543") {
            Err(PairParseError(PairParseErrorType::ValueParseError)) => {}
            other => panic!("Error expected. Got {:?}", other),
        }
    }
}

#[cfg(test)]
mod pixel_to_point_tests {
    use super::*;

    #[test]
    fn converts_pixel_to_complex_point() {
        let result = pixel_to_point(
            (100, 200),
            (25, 175),
            Complex { re: -1.0, im: 1.0 },
            Complex { re: 1.0, im: -1.0 },
        );

        assert_eq!(
            result,
            Complex {
                re: -0.5,
                im: -0.75,
            }
        );
    }
}
