use std::ops::{Index, IndexMut};

use make_it_braille::BrailleImg;
use utils::math::sigmoid;

#[derive(Debug, Clone)]
pub struct Vector1D(Vec<f64>);
impl Vector1D {
	pub fn with_capacity(capacity: usize) -> Self {
		Self(Vec::with_capacity(capacity))
	}

	pub fn dot(&self, other: &Vector1D) -> f64 {
		self.iter().zip(other.iter())
			.map(|(a, b)| a * b)
			.sum()
	}

	pub fn iter(&self) -> std::slice::Iter<f64> {
		self.0.iter()
	}

	pub fn len(&self) -> usize {
		self.0.len()
	}

	pub fn copy_from_slice(&mut self, slice: &[f64]) {
		self.0.copy_from_slice(slice);
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

	pub fn mul(&self, rhs: f64) -> Self {
		Self(self.iter().map(|x| x * rhs).collect())
	}

	pub fn sub(&self, rhs: &Self) -> Self {
		Self(self.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect())
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

fn draw_image(img: &mut BrailleImg, pixels: &Vec<f64>, x: u32, y: u32) {
	pixels.chunks(28).enumerate().for_each(|(y1, rows)| {
		rows.iter().enumerate().for_each(|(x1, val)| {
			img.set_dot(x + x1 as u32, y + y1 as u32, *val > 0.5)
				.expect("Error setting dot");
		});
	});
}

pub fn sigmoid_prime(z: f64) -> f64 {
	sigmoid(z) * (1.0 - sigmoid(z))
}
