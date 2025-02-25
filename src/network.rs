// Implements the Neural Network module
extern crate rand;

use chrono::Local;
use rand::prelude::*;
use rand::seq::SliceRandom;
use utils::math::sigmoid;

use crate::utils::WrongAnswers;

#[derive(Savefile)]
pub struct Network {
	pub sizes: Vec<u32>,
	pub num_layers: u32,
	pub biases: Vec<Vec<f64>>,
	pub weights: Vec<Vec<Vec<f64>>>,
}

impl Network {
	// Constructor
	pub fn new(sizes_arg: Vec<u32>) -> Self {
		Self {
			sizes: sizes_arg.clone(),
			num_layers: sizes_arg.len() as u32,
			biases: Self::form_biases(&sizes_arg),
			weights: Self::form_weights(&sizes_arg),
		}
	}

	fn form_biases(in_sizes: &Vec<u32>) -> Vec<Vec<f64>> {
		// Rust version of normal standard distribution as used in the mnist tutorial
		// in numpy one just takes y and z standard normal distributions and puts them in an array
		// Code partially from here https://docs.rs/rand/0.6.5/rand/distributions/struct.Normal.html

		let mut rng = rand::rng();
		let mut bs: Vec<Vec<f64>> = vec![];
		for size in in_sizes[1..].iter() {
			let mut res: Vec<f64> = vec![];

			for _ in 0..*size {
				res.push(rng.random_range(-1.0..1.0));
			}

			bs.push(res);
		}
		bs
	}

	fn form_weights(in_sizes: &Vec<u32>) -> Vec<Vec<Vec<f64>>> {
		// Also uses standard distribution but uses zip to aggregate
		// two separate slices of the input.
		// Returns triple dimensional Vector with [sec/third_layers[]]

		let mut weights = vec![];

		let locator = in_sizes.len() - 1;
		let size_iter = in_sizes[1..].iter().zip(in_sizes[..locator].iter());

		let mut rng = rand::rng();
		for (x, y) in size_iter {
			let mut overholder = vec![];

			for _ in 0..*x {
				let mut placeholder = vec![];
				for _ in 0..*y {
					placeholder.push(rng.random_range(-1.0..1.0));
				}

				overholder.push(placeholder);
			}
			weights.push(overholder);
		}

		weights
	}

	pub fn sgd(
		&mut self,
		training_data: &Vec<Vec<Vec<f64>>>,
		epochs: usize,
		success_percentage: f64,
		mini_batch_size: usize,
		eta: f64,
		test_data: &Vec<(Vec<f64>, i32)>,
	) {
		// Print evaluation result
		let (correct_images, _) = self.evaluate(test_data);
		println!("Before Epoch: {} / {}", correct_images, test_data.len());

		let mut td = training_data.clone();

		let mut idx = 0;
		let mut accuracy: f64 = 0.0;
		let mut rng = rand::rng();
		while idx <= epochs && accuracy < success_percentage {
			let t = Local::now();

			// Shuffle and slice training data by mini_batch_size
			td.shuffle(&mut rng);

			//// Apply backpropagation on mini_batches
			for mb in td.chunks(mini_batch_size) {
				self.update_mini_batch(mb, eta);
			}

			// Print evaluation result
			let (correct_images, wrong_answers) = self.evaluate(test_data);
			accuracy = correct_images as f64 / test_data.len() as f64;

			//wrong_answers.dump();
			println!("Epoch {}: {} / {} ({:.1}%), ({}s delayed)", idx, correct_images, test_data.len(), accuracy * 100.0, (Local::now() - t).num_seconds());

			//println!("weights are: {:?}", self.weights[0][0]);
			idx = idx + 1;
		}
	}

	// Takes and modifies an input image-vector and returns results
	// based on previously defined weights.
	pub fn feedforward(&self, a: &Vec<f64>) -> Vec<f64> {
		let feedforward_for_layer = |a: Vec<f64>, (b, w): (&Vec<f64>, &Vec<Vec<f64>>)| {
			sigmoid_vec(&add_vec(&dot(&a, w), b))
		};
		self.biases.iter().zip(self.weights.iter())
			.fold(a.clone(), feedforward_for_layer)
	}

	fn backpropagate(&self, x: &Vec<f64>, y: &Vec<f64>) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
		// Calculates the gradient for the cost-function

		// Initialise gradient vectors:
		let mut nabla_b = self.biases.iter()
			.map(|bias| vec![0.; bias.len()])
			.collect::<Vec<Vec<f64>>>();
		let mut nabla_w = self.weights.iter()
			.map(|weight| weight.iter()
				.map(|w| vec![0.; w.len()])
				.collect())
			.collect::<Vec<Vec<Vec<f64>>>>();

		// Feedforward part (according to template)
		// I use clones mostly to not get confused about ownership at this stage
		let mut activation = x.clone();
		let mut activations = vec![activation.clone()];

		let mut zs = vec![];
		for (bias, weight) in self.biases.clone().iter().zip(self.weights.clone().iter()) {
			let z = add_vec(&dot(&activation.clone(), weight), bias);
			//println!("LEN CHECK z, bias, weight/act: {:?} {:?} {:?}", z.len(), bias.len(), &dot(&activation, weight).len());

			zs.push(z.clone());

			// Compute activation of next layer with sigmoid
			activation = sigmoid_vec(&z);
			activations.push(activation.clone());
		}

		// Backward pass
		// Compute error/delta vector and backpropagate the error
		let mut delta = dot_1d(
			&self.cost_derivative(&activations[activations.len() - 1], y),
			&sigmoid_prime(&zs[zs.len() - 1]),
		);
		let mut locator = nabla_b.len() - 1;
		nabla_b[locator] = delta.clone();

		//println!("Das ist delta god damnit: {:?}", &delta);
		//println!("Und das ist nabla_b: {:?}", &nabla_b);

		locator = nabla_w.len() - 1;
		// b input of Activations was transposed in python, but doesn't need to be
		// with current data structure
		nabla_w[locator] = dot_2d_from_1d(&delta, &activations[activations.len() - 2]);
		//println!("Das ist nabla_w: {:?}", nabla_w);

		for i in 2..self.num_layers {
			// i is used to measure the distance from the end
			// of the zs-vector

			locator = zs.len() - i as usize;

			let z = &zs[locator];
			let sp = sigmoid_prime(&z);

			locator = self.weights.len() - i as usize;
			//locator = (self.weights.len() as f64 - (i + 1) as f64) as usize;
			//println!("THESE ARE WEIGHTS: \n {:?}", self.weights);
			let weight_fodder = transpose(self.weights[locator].clone());
			let temp_delta = dot(&delta, &weight_fodder);

			delta = dot_1d(&temp_delta, &sp);

			// Updating of gradients
			locator = nabla_b.len() - i as usize;
			nabla_b[locator] = delta.clone();

			locator = nabla_w.len() - i as usize;
			let activations_loc = activations.len() - (i + 1) as usize;
			nabla_w[locator] = dot_2d_from_1d(&delta, &activations[activations_loc]);
		}

		(nabla_b, nabla_w)
	}

	fn update_mini_batch(&mut self, mini_batch: &[Vec<Vec<f64>>], eta: f64) {
		// Used to update weights and biases through a single gradient
		// descent iteration

		// Initialise gradient vectors
		//println!("WEIGHTS 10: {:?}", &self.weights[0][0][..10]);

		let mut nabla_b: Vec<Vec<f64>> = Vec::new();

		for bias in &self.biases {
			nabla_b.push(vec![0.; bias.len()]);
		}

		let mut nabla_w: Vec<Vec<Vec<f64>>> = Vec::new();

		for weight in &self.weights {
			let mut w: Vec<Vec<f64>> = Vec::new();
			for ws in weight {
				w.push(vec![0.; ws.len()]);
			}
			nabla_w.push(w)
		}

		//println!("MINIB {:?} \n\n\n\n\n", mini_batch[0]);

		for mb in mini_batch {
			// Compute gradients of cost function with backpropagation
			assert_eq!(mb.len(), 2);
			let (delta_nabla_b, delta_nabla_w) = self.backpropagate(&mb[0], &mb[1]);
			//println!("NABLA_W FIRST 10: {:?}", delta_nabla_w);
			// Update gradients
			let mut placeholder_nb: Vec<Vec<f64>> = Vec::new();
			let mut placeholder_nw: Vec<Vec<Vec<f64>>> = Vec::new();

			for (nb, dnb) in nabla_b.iter().zip(&delta_nabla_b) {
				placeholder_nb.push(add_vec(nb, dnb));
			}

			nabla_b = placeholder_nb;

			for (nw_2d, dnw_2d) in nabla_w.iter().zip(delta_nabla_w) {
				let mut ph: Vec<Vec<f64>> = Vec::new();
				for (x, y) in nw_2d.iter().zip(dnw_2d) {
					ph.push(add_vec(x, &y));
				}
				placeholder_nw.push(ph);
			}

			nabla_w = placeholder_nw;
		}

		let m = mini_batch.len();
		//println!("biases: {:?}", self.biases);
		//println!("nabla_b: {:?}", nabla_b);
		let ph_b: Vec<Vec<f64>> = self.biases.iter().zip(&nabla_b)
			.map(|(b, nb)| {
				b.iter().zip(nb)
					.map(|(bs, nbs)| bs - nbs * eta / m as f64)
					.collect()
			})
			.collect();

		//println!("weights: {:?}", self.weights[0][0][200]);
		//println!("nabla_w: {:?}", nabla_w[0][0][200]);
		let ph_w: Vec<Vec<Vec<f64>>> = self.weights.iter().zip(&nabla_w)
			.map(|(w, nw)| {
				w.iter().zip(nw)
					.map(|(min_w, min_nw)| {
						min_w.iter().zip(min_nw)
							.map(|(ws, nws)| ws - nws * eta / m as f64)
							.collect()
					})
					.collect()
			})
			.collect();

		self.biases = ph_b;
		self.weights = ph_w;
		//println!("Weights are: {:?}", self.weights[0][5]);
	}

	fn cost_derivative(&self, output_activations: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
		// Could be solved without the name specialization as simple
		// subtraction
		assert_eq!(&output_activations.len(), &y.len());

		let mut prod = vec![];
		for i in 0..output_activations.len() {
			prod.push(output_activations[i] - y[i]);
		}
		prod
	}

	fn evaluate<'a>(&self, test_data: &'a Vec<(Vec<f64>, i32)>) -> (i32, WrongAnswers<'a>) {
		// Returns the sum of correctly assigned test inputs.
		let mut wrongs = WrongAnswers::new();
		let r = test_data.iter()
			.map(|(x, y)| {
				let r = argmax(&self.feedforward(x)) as i32;
				if r == *y {
					1
				} else {
					wrongs.push((x, *y, r));
					0
				}
			})
			.sum();
		(r, wrongs)
	}
}

// Emulates the numpy.dot() function which takes and returns
// the product of two matrices. In this case vectors are used.
fn dot(a: &Vec<f64>, matrix: &Vec<Vec<f64>>) -> Vec<f64> {
	matrix.iter()
		.map(|row| {
			row.iter().zip(a.iter())
				.map(|(mi, ai)| mi * ai)
				.sum()
		})
		.collect()
}

// Calculates only the product of 2 1Dimensional Vectors.
fn dot_1d(a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
	b.iter().zip(a.iter())
		.map(|(x, y)| x * y)
		.collect()
}

// This function seemed neccessary since the numpy version of
// the dot function differed in outcome from my own with uneven
// inputs (in this case delta and activation)
// which returned a 2d array in python
fn dot_2d_from_1d(a: &Vec<f64>, b: &Vec<f64>) -> Vec<Vec<f64>> {
	a.iter()
		.map(|ai| b.iter().map(|bj| ai * bj).collect())
		.collect()
}

// Simply adds the single values of 2 similarly sized vectors.
// Mapping function would be too difficult in this case.
fn add_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
	assert_eq!(a.len(), b.len());
	a.iter()
		.zip(b.iter())
		.map(|(x, y)| x + y)
		.collect()
}

pub fn argmax(a: &[f64]) -> usize {
	a.iter()
		.enumerate()
		.max_by(|(_, a), (_, b)| a.total_cmp(b))
		.map(|(i, _)| i)
		.unwrap_or(0)
}

// Following function was taken from here:
// https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
	T: Clone,
{
	assert!(!v.is_empty());
	(0..v[0].len())
		.map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
		.collect()
}

fn sigmoid_vec(a: &[f64]) -> Vec<f64> {
	// Calculates the sigmoid of input array

	a.iter().map(|x| sigmoid(*x)).collect()
}

fn sigmoid_prime(a: &[f64]) -> Vec<f64> {
	// I'm sorry for this mess
	// Basically it gets the product from the sigmoid calculated vector
	// "sigmoid_val" and adds a -1 vector and multiplies the product
	// with itself (sigmoid_val) and returns the product.

	a.iter()
		.map(|&v| {
			let v = sigmoid(v);
			v * (1. - v)
		})
		.collect()
}
