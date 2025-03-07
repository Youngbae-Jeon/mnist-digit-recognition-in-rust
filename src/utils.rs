use make_it_braille::BrailleImg;
use rand::Rng;
use rand_distr::StandardNormal;

use crate::matrix::Matrix;

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

pub struct VecInitializer {
	_f: fn(size: usize) -> Vec<f64>,
}
impl VecInitializer {
	pub const STANDARD_NORMAL_SQRT: VecInitializer = VecInitializer {
		_f: |weights_len| {
			let mut rng = rand::rng();
			let random = |_| rng.sample::<f64,_>(StandardNormal) / (weights_len as f64).sqrt();
			(0..weights_len).map(random).collect()
		},
	};
	pub const STANDARD_NORMAL: VecInitializer = VecInitializer {
		_f: |weights_len| {
			let mut rng = rand::rng();
			let random = |_| rng.sample::<f64,_>(StandardNormal);
			(0..weights_len).map(random).collect()
		},
	};

	#[inline]
	pub fn f(&self, weights_len: usize) -> Vec<f64> {
		(self._f)(weights_len)
	}
}


#[derive(Clone, Copy)]
pub struct WeightInitializer {
	_f: fn(weights_len: usize) -> Vec<f64>,
}
impl WeightInitializer {
	/// Initialize each weight using a Gaussian distribution with mean 0
	/// and standard deviation 1 over the square root of the number of
	/// weights connecting to the same neuron.  Initialize the biases
	/// using a Gaussian distribution with mean 0 and standard
	/// deviation 1.
	///
	/// Note that the first layer is assumed to be an input layer, and
	/// by convention we won't set any biases for those neurons, since
	/// biases are only ever used in computing the outputs from later
	/// layers.
	pub const DEFAULT: WeightInitializer = WeightInitializer {
		_f: VecInitializer::STANDARD_NORMAL_SQRT._f,
	};

	/// Initialize the weights using a Gaussian distribution with mean 0
	/// and standard deviation 1.  Initialize the biases using a
	/// Gaussian distribution with mean 0 and standard deviation 1.
	///
	/// Note that the first layer is assumed to be an input layer, and
	/// by convention we won't set any biases for those neurons, since
	/// biases are only ever used in computing the outputs from later
	/// layers.
	///
	/// This weight and bias initializer uses the same approach as in
	/// Chapter 1, and is included for purposes of comparison.  It
	/// will usually be better to use the default weight initializer
	/// instead.
	pub const LARGE: WeightInitializer = WeightInitializer {
		_f: VecInitializer::STANDARD_NORMAL._f,
	};

	#[inline]
	pub fn f(&self, weights_len: usize) -> Vec<f64> {
		(self._f)(weights_len)
	}
}
impl Default for WeightInitializer {
	fn default() -> Self {
		Self::DEFAULT
	}
}

pub fn sigmoid(o: f64) -> f64 {
	1.0 / (1.0 + (-o).exp())
}

pub fn sigmoid_prime(o: f64) -> f64 {
	sigmoid(o) * (1.0 - sigmoid(o))
}

#[derive(Clone, Copy)]
pub struct ActivationFunction {
	pub _f: fn(z: f64) -> f64,
	pub _prime: fn(z: f64) -> f64,
}
impl ActivationFunction {
	#[allow(dead_code)]
	pub const SIGMOID: ActivationFunction = ActivationFunction {
		_f: sigmoid,
		_prime: sigmoid_prime,
	};
	#[allow(dead_code)]
	pub const NOOP: ActivationFunction = ActivationFunction {
		_f: |z| z,
		_prime: |_| 1.0,
	};
	#[inline]
	pub fn f(&self, z: f64) -> f64 {
		(self._f)(z)
	}
	#[inline]
	pub fn prime(&self, z: f64) -> f64 {
		(self._prime)(z)
	}
}

#[derive(Clone, Copy)]
pub struct CostFunction {
	pub _f: fn(a: &Matrix, y: &Matrix) -> f64,
	pub _delta: fn(z: &Matrix, a: &Matrix, y: &Matrix) -> Matrix,
}
impl CostFunction {
	#[allow(dead_code)]
	pub const QUADRATIC: CostFunction = CostFunction {
		_f: |a, y| {
			assert_eq!(a.shape, y.shape);
			Matrix::update(a.clone() - y, |x| x.powi(2))
				.iter()
				.sum::<f64>()
		},
		_delta: |z, a, y| {
			assert_eq!(z.shape, a.shape);
			assert_eq!(a.shape, y.shape);
			assert_eq!(a.shape.1, 1);
			let mut delta = vec![0.0; a.shape.0];
			let a = a.column(0);
			let y = y.column(0);
			let z = z.column(0);
			for i in 0..a.len() {
				delta[i] = (a[i] - y[i]) * sigmoid_prime(z[i]);
			}
			Matrix::from(delta)
		}
	};
	#[allow(dead_code)]
	pub const CROSS_ENTROPY: CostFunction = CostFunction {
		_f: |a, y| {
			assert_eq!(a.shape, y.shape);
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
		_delta: |_z, a, y| a.clone() - y,
	};
	pub fn f(&self, a: &Matrix, y: &Matrix) -> f64 {
		(self._f)(a, y)
	}
	pub fn delta(&self, z: &Matrix, a: &Matrix, y: &Matrix) -> Matrix {
		(self._delta)(z, a, y)
	}
}
impl Default for CostFunction {
	fn default() -> Self {
		Self::CROSS_ENTROPY
	}
}

#[derive(Default)]
pub struct TrainingOptions {
	pub epochs: usize,
	pub success_percentage: f64,
	pub mini_batch_size: usize,
	pub eta: f64,
	pub lambda: f64,
}
