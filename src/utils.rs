use std::{fmt::{self, Display, Formatter}, iter::FromIterator, ops::{AddAssign, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign}};

use make_it_braille::BrailleImg;
use rand::Rng;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct Vector1D(Vec<f64>);
impl Vector1D {
	pub fn zero(size: usize) -> Self {
		Self(vec![0.0; size])
	}

	pub fn dot(&self, other: &Vector1D) -> f64 {
		assert_eq!(self.len(), other.len());
		self.iter().zip(other.iter())
			.map(|(a, b)| a * b)
			.sum()
	}

	pub fn iter(&self) -> std::slice::Iter<f64> {
		self.0.iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
		self.0.iter_mut()
	}

	pub fn len(&self) -> usize {
		self.0.len()
	}

	pub fn copy_from_slice(&mut self, slice: &[f64]) {
		self.0.copy_from_slice(slice);
	}

	pub fn fill(&mut self, val: f64) {
		self.0.fill(val);
	}

	pub fn find_max(&self) -> (f64, usize) {
		self.iter().enumerate()
			.fold((0.0, 0), |(max_val, max_idx), (idx, &val)| {
				if val > max_val {
					(val, idx)
				} else {
					(max_val, max_idx)
				}
			})
	}
}
impl From<Vec<f64>> for Vector1D {
	fn from(vec: Vec<f64>) -> Self {
		Self(vec)
	}
}
impl Index<usize> for Vector1D {
	type Output = f64;

	fn index(&self, index: usize) -> &Self::Output {
		&self.0[index]
	}
}
impl IndexMut<usize> for Vector1D {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		&mut self.0[index]
	}
}
impl AddAssign<&Vector1D> for Vector1D {
	fn add_assign(&mut self, rhs: &Vector1D) {
		assert_eq!(self.len(), rhs.len());
		self.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a += b);
	}
}
impl SubAssign<&Vector1D> for Vector1D {
	fn sub_assign(&mut self, rhs: &Vector1D) {
		assert_eq!(self.len(), rhs.len());
		self.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a -= b);
	}
}
impl Sub<&Vector1D> for Vector1D {
	type Output = Self;

	fn sub(mut self, rhs: &Vector1D) -> Self::Output {
		assert_eq!(self.len(), rhs.len());
		self.sub_assign(rhs);
		self
	}
}
impl DivAssign<f64> for Vector1D {
	fn div_assign(&mut self, rhs: f64) {
		self.iter_mut().for_each(|x| *x /= rhs);
	}
}
impl MulAssign<f64> for Vector1D {
	fn mul_assign(&mut self, rhs: f64) {
		self.iter_mut().for_each(|x| *x *= rhs);
	}
}
impl Mul<f64> for Vector1D {
	type Output = Self;

	fn mul(mut self, rhs: f64) -> Self::Output {
		self.mul_assign(rhs);
		self
	}
}
impl MulAssign<&Vector1D> for Vector1D {
	fn mul_assign(&mut self, rhs: &Vector1D) {
		assert_eq!(self.len(), rhs.len());
		self.iter_mut().zip(rhs.iter()).for_each(|(a, b)| *a *= b);
	}
}
impl Mul<&Vector1D> for Vector1D {
	type Output = Vector1D;

	fn mul(mut self, rhs: &Vector1D) -> Self::Output {
		self.mul_assign(rhs);
		self
	}
}
impl Display for Vector1D {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		write!(f, "Vector1D(len={})", self.len())
	}
}
impl FromIterator<f64> for Vector1D {
	fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
		Self(Vec::from_iter(iter))
	}
}

const IMAGE_WIDTH: usize = 28;

type WrongAnswer<'a> = (&'a Vec<f64>, i32, i32);

pub struct WrongAnswers<'a> {
	answers: Vec<WrongAnswer<'a>>,
}
impl<'a> WrongAnswers<'a> {
	pub fn new() -> Self {
		Self {
			answers: Vec::new(),
		}
	}

	pub fn push(&mut self, answer: WrongAnswer<'a>) {
		self.answers.push(answer);
	}
	
	pub fn dump(&self) {
		self.answers.chunks(5).for_each(dump_wrong_chunk);
		println!("Wrong guesses: {}", self.answers.len());
	}
}

fn dump_wrong_chunk(chunk: &[(&Vec<f64>, i32, i32)]) {
	let mut img = BrailleImg::new(28 * chunk.len() as u32, 28);
	let mut labels: String = String::new();
	let mut guesses: String = String::new();
	chunk.iter().enumerate().for_each(|(i, (pixels, label, guess))| {
		draw_image(&mut img, pixels, (i * IMAGE_WIDTH) as u32, 0);
		labels.push_str(&format!(" Label: {:?}      ", label));
		guesses.push_str(&format!("(Guess: {:?})     ", guess));
	});
	println!("{}", img.as_str(false, true));
	println!("{}", labels);
	println!("{}", guesses);
}

fn draw_image(img: &mut BrailleImg, pixels: &[f64], x: u32, y: u32) {
	pixels.chunks(28).enumerate().for_each(|(y1, rows)| {
		rows.iter().enumerate().for_each(|(x1, val)| {
			img.set_dot(x + x1 as u32, y + y1 as u32, *val > 0.5)
				.expect("Error setting dot");
		});
	});
}

#[derive(Clone, Copy)]
pub struct WeightsInitializer {
	pub f: fn(weights_len: usize) -> Vector1D,
}
impl WeightsInitializer {
	pub const DEFAULT: WeightsInitializer = WeightsInitializer {
		f: |weights_len| {
			let mut rng = rand::rng();
			let random = |_| rng.sample::<f64,_>(StandardNormal) / (weights_len as f64).sqrt();
			(0..weights_len).map(random).collect()
		},
	};
	pub const LARGE: WeightsInitializer = WeightsInitializer {
		f: |weights_len| {
			let mut rng = rand::rng();
			let random = |_| rng.sample::<f64,_>(StandardNormal);
			(0..weights_len).map(random).collect()
		},
	};
}

pub fn sigmoid(o: f64) -> f64 {
	1.0 / (1.0 + (-o).exp())
}

pub fn sigmoid_prime(o: f64) -> f64 {
	sigmoid(o) * (1.0 - sigmoid(o))
}

#[derive(Clone, Copy)]
pub struct ActivationFunction {
	pub f: fn(z: f64) -> f64,
	pub prime: fn(z: f64) -> f64,
}
impl ActivationFunction {
	pub const SIGMOID: ActivationFunction = ActivationFunction {
		f: sigmoid,
		prime: sigmoid_prime,
	};
	pub const NOOP: ActivationFunction = ActivationFunction {
		f: |z| z,
		prime: |_| 1.0,
	};
}

#[derive(Clone, Copy)]
pub struct CostFunction {
	pub f: fn(a: &Vector1D, y: &Vector1D) -> f64,
	pub derivative: fn(a: &Vector1D, y: &Vector1D) -> Vector1D,
}
impl CostFunction {
	pub const QUADRATIC: CostFunction = CostFunction {
		f: |a, y| {
			assert_eq!(a.len(), y.len());
			0.5 * a.iter().zip(y.iter())
				.map(|(a, y)| (a - y).powi(2))
				.sum::<f64>()
		},
		derivative: |a, y| a.clone() - y,
	};
	pub const CROSS_ENTROPY: CostFunction = CostFunction {
		f: |a, y| {
			assert_eq!(a.len(), y.len());
			a.iter().zip(y.iter())
				.map(|(a, y)| {
					if a - y == 0.0 {
						0.0
					} else {
						-y * a.ln() - (1.0 - y) * (1.0 - a).ln()
					}
				})
				.sum::<f64>()
		},
		derivative: |a, y| {
			assert_eq!(a.len(), y.len());
			a.iter().zip(y.iter())
				.map(|(&a, &y)| {
					if a == 1.0 {
						- y / a
					} else if a == 0.0 {
						(1.0 - y) / (1.0 - a)
					} else {
						- y / a + (1.0 - y) / (1.0 - a)
					}
				})
				.collect()
		}
	};
}

pub struct TrainingOptions {
	pub epochs: usize,
	pub success_percentage: f64,
	pub mini_batch_size: usize,
	pub eta: f64,
	pub lambda: f64,
}
