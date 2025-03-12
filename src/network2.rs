// ported from network2.py
// ----------------------
// An improved version of network.py, implementing the stochastic
// gradient descent learning algorithm for a feedforward neural network.
// Improvements include the addition of the cross-entropy cost function,
// regularization, and better initialization of network weights.  Note
// that I have focused on making the code simple, easily readable, and
// easily modifiable.  It is not optimized, and omits many desirable
// features.

use chrono::Local;
use ndarray::ArrayView1;
use rand::seq::SliceRandom;

use crate::{io::{vectorise_num, TrainData}, utils::{ActivationFunction, CostFunction, TrainingOptions, VecInitializer, WeightInitializer}};

// network2.rs uses `ndarray::Array2` as Matrix instead of inhouse `matrix::Matrix`.
// `ndarray` uses cpu features like AVX or SSE2 or BLAS
// and makes faster performance than the inhouse matrix implementation.
type Matrix = ndarray::Array2<f64>;

#[derive(Clone)]
struct Layer {
	/// matrix of shape (neurons_len, weights_len)
	weights: Matrix,
	/// matrix of shape (neurons_len, 1)
	biases: Matrix,
}
impl Layer {
	fn size(&self) -> usize {
		self.biases.nrows()
	}
}

#[derive(Default)]
pub struct NetworkOptions {
	pub shape: Vec<usize>,
	pub weights_initializer: WeightInitializer,
	pub cost: CostFunction,
}

pub struct Network {
	shape: Vec<usize>,
	layers: Vec<Layer>,
	cost_fn: CostFunction,
}

impl Network {
	/// The list ``shape`` contains the number of neurons in the respective
	/// layers of the network.  For example, if the list was [2, 3, 1]
	/// then it would be a three-layer network, with the first layer
	/// containing 2 neurons, the second layer 3 neurons, and the
	/// third layer 1 neuron.  The weights for the network
	/// are initialized randomly, using
	/// ``WeightInitializer::DEFAULT`` (see docstring for that method).
	/// The biases for the network are initialized randomly, using
	/// a Gaussian distribution with mean 0, and variance 1.
	/// (aka. Standard Normal Distribution)
	pub fn new (options: &NetworkOptions) -> Self {
		Self {
			shape: options.shape.to_vec(),
			layers: options.shape.windows(2)
				.map(|w| {
					let weights_len = w[0];
					let neurons_len = w[1];
					let weights_data = options.weights_initializer.f(weights_len * neurons_len);
					let biases_data: Vec<f64> = VecInitializer::STANDARD_NORMAL.f(neurons_len);
					Layer {
						weights: Matrix::from_shape_vec((neurons_len, weights_len), weights_data).unwrap(),
						biases: Matrix::from_shape_vec((neurons_len, 1), biases_data).unwrap(),
					}
				})
				.collect(),
			cost_fn: options.cost.clone(),
		}
	}

	/// Return the output of the network if ``x`` is input.
	fn feedforward(&self, x: &Matrix) -> Matrix {
		let a = self.layers.iter()
			.fold(x.clone(), |x, layer| {
				assert_eq!(x.nrows(), layer.weights.ncols());
				let z = layer.weights.dot(&x) + &layer.biases;
				let a = self.activation(z);
				assert_eq!(a.nrows(), layer.size());
				assert_eq!(a.ncols(), 1);
				a
			});
		assert_eq!(a.nrows(), 10);
		assert_eq!(a.ncols(), 1);
		a
	}

	/// Train the neural network using mini-batch stochastic gradient
	/// descent.  The ``training_data`` is a list of tuples ``(x, y)``
	/// representing the training inputs and the desired outputs.  The
	/// other non-optional parameters are self-explanatory, as is the
	/// regularization parameter ``lmbda``.
	pub fn sgd(
		&mut self,
		options: &TrainingOptions,
		data: &TrainData,
	) {
		if options.lambda < 0.0 {
			println!("Lambda must be non-negative. Negative lambda will be ignored.");
		}

		// change data format to tuples of (Matrix, Matrix) representing (x, y)
		let mut training_data: Vec<(Matrix, Matrix)> = data.training_data.iter()
			.map(|arr| {
				assert_eq!(arr.len(), 2);
				assert_eq!(arr[0].len(), 784);
				assert_eq!(arr[1].len(), 10);
				let x = Matrix::from_shape_vec((784, 1), arr[0].clone()).unwrap();
				let y = Matrix::from_shape_vec((10, 1), arr[1].clone()).unwrap();
				(x, y)
			})
			.collect();
		let validation_data: Vec<(Matrix, Matrix)> = data.validation_data.iter()
			.map(|(x, y)| {
				assert_eq!(x.len(), 784);
				let x = Matrix::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
				let y = Matrix::from_shape_vec((10, 1), vectorise_num(&(*y as u8))).unwrap();
				(x, y)
			})
			.collect();

		println!("Training with {} data...", training_data.len());
		let (score, cost) = self.evaluate(&validation_data, options.lambda);
		let accuracy = score as f64 / validation_data.len() as f64;
		println!(" Epoch |  Train   | Train  | Validation | Validation | Elapsed ");
		println!("       | Accuracy |  Cost  |  Accuracy  |    Cost    |   Time  ");
		println!("-------|----------|--------|------------|------------|---------");
		println!(" {:>5} | {:>8} | {:>6} | {:>9.2}% | {:>10.3} | {:>7}", 0, "-", "-", accuracy * 100.0, cost, "-");

		let mut best_layers: Vec<Layer> = self.layers.clone();
		let mut best_accu = accuracy;
		let mut best_epoch = 0;

		let mut rng = rand::rng();
		for epoch in 0..options.epochs {
			let t = Local::now();
			training_data.shuffle(&mut rng);

			for mini_batch in training_data.chunks(options.mini_batch_size) {
				self.update_mini_batch(mini_batch, options.eta, options.lambda, training_data.len());
			}

			let (train_score, train_cost) = self.evaluate(&training_data, options.lambda);
			let train_accu = train_score as f64 / training_data.len() as f64;

			let (validation_score, validation_cost) = self.evaluate(&validation_data, options.lambda);
			let validation_accu = validation_score as f64 / validation_data.len() as f64;
			let elapsed = (Local::now() - t).num_milliseconds() as f64 / 1000.0;
			println!(" {:>5} | {:>7.2}% | {:>6.3} | {:>9.2}% | {:>10.3} | {:>6.1}s",
				epoch+1, train_accu*100.0, train_cost, validation_accu*100.0, validation_cost, elapsed);

			if validation_accu > best_accu {
				best_layers = self.layers.clone();
				best_accu = validation_accu;
				best_epoch = epoch;
			};
		}

		self.layers = best_layers;
		let test_data: Vec<(Matrix, Matrix)> = data.test_data.iter()
			.map(|(x, y)| {
				assert_eq!(x.len(), 784);
				let x = Matrix::from_shape_vec((x.len(), 1), x.to_vec()).unwrap();
				let y = Matrix::from_shape_vec((10, 1), vectorise_num(&(*y as u8))).unwrap();
				(x, y)
			})
			.collect();
		let (test_score, test_cost) = self.evaluate(&test_data, options.lambda);
		let test_accu = test_score as f64 / test_data.len() as f64;
		println!("---------------------------|------------|------------|---------");
		println!(" {:25} | {:>9.2}% | {:>10.3} | {:>7}", format!("Test by Best Epoch {}", best_epoch+1), test_accu * 100.0, test_cost, "");
	}

	#[cfg(not(feature = "rayon"))]
	fn mini_batch_nabla(&self, mini_batch: &[(Matrix, Matrix)]) -> Vec<Layer> {
		let mut nabla: Vec<Layer> = self.layers.iter()
			.map(|layer| Layer {
				weights: Matrix::zeros((layer.weights.nrows(), layer.weights.ncols())),
				biases: Matrix::zeros((layer.biases.nrows(), layer.biases.ncols())),
			})
			.collect();
		for (x, y) in mini_batch {
			let delta_nabla = self.backprop(x, y);
			assert_eq!(delta_nabla.len(), nabla.len());
			nabla.iter_mut().zip(delta_nabla.into_iter())
				.for_each(|(n, dn)| {
					n.weights += &dn.weights;
					n.biases += &dn.biases;
				});
		}
		nabla
	}

	#[cfg(feature = "rayon")]
	fn mini_batch_nabla(&self, mini_batch: &[(Matrix, Matrix)]) -> Vec<Layer> {
		use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

		mini_batch.par_iter()
			.map(|(x, y)| {
				let delta_nabla = self.backprop(x, y);
				delta_nabla
			})
			.reduce(|| {
				self.layers.iter()
					.map(|layer| Layer {
						weights: Matrix::zeros((layer.weights.nrows(), layer.weights.ncols())),
						biases: Matrix::zeros((layer.biases.nrows(), layer.biases.ncols())),
					})
					.collect::<Vec<_>>()
			}, |mut nabla, delta_nabla| {
				assert_eq!(delta_nabla.len(), nabla.len());
				nabla.iter_mut().zip(delta_nabla.into_iter())
					.for_each(|(n, dn)| {
						n.weights += &dn.weights;
						n.biases += &dn.biases;
					});
				nabla
			})
	}

	/// Update the network's weights and biases by applying gradient
	/// descent using backpropagation to a single mini batch.  The
	/// ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
	/// learning rate, ``lambda`` is the regularization parameter, and
	/// ``n`` is the total size of the training data set.
	pub fn update_mini_batch(&mut self, mini_batch: &[(Matrix, Matrix)], eta: f64, lambda: f64, n: usize) {
		let nabla = self.mini_batch_nabla(mini_batch);

		let m = mini_batch.len() as f64;
		let n = n as f64;
		self.layers.iter_mut().zip(nabla)
			.for_each(|(layer, nabla)| {
				if lambda > 0.0 {
					layer.weights *= 1.0 - eta * (lambda/n); // L2 regularization
				}
				layer.weights -= &(nabla.weights * (eta/m));
				layer.biases -= &(nabla.biases * (eta/m));
			});
	}

	/// Return a struct ``Layer { weights: nabla_w, biases: nabla_b }`` representing the
	/// gradient for the cost function C_x.
	fn backprop(&self, x: &Matrix, y: &Matrix) -> Vec<Layer> {
		let mut reversed_nabla: Vec<Layer> = Vec::with_capacity(self.layers.len());

		// feedforward
		let mut activations = Vec::with_capacity(self.layers.len() + 1); // list to store all the activations, layer by layer
		activations.push(x.clone());
		let mut zs: Vec<Matrix> = Vec::with_capacity(self.layers.len()); // list to store all the z vectors, layer by layer

		for layer in self.layers.iter() {
			let x = activations.last().unwrap();
			let z = layer.weights.dot(x) + &layer.biases;
			zs.push(z.clone());

			let a = self.activation(z);
			activations.push(a);
		}

		// backward pass
		//let cost_derivative = self.cost_derivative(activations.last().unwrap().clone(), y);
		//let sp = self.activation_prime(zs.last().unwrap().clone());
		//assert_eq!(cost_derivative.shape, sp.shape);

		//let mut delta = cost_derivative * sp;
		let mut delta = self.delta(zs.last().unwrap(), &activations.last().unwrap(), y);
		reversed_nabla.push(Layer {
			weights: delta.dot(&activations[activations.len() - 2].t()),
			biases: delta.clone(),
		});

		// Note that the variable l in the loop below is used a little
		// differently to the notation in Chapter 2 of the book.  Here,
		// l = 1 means the last layer of neurons, l = 2 is the
		// second-last layer, and so on.  It's a renumbering of the
		// scheme in the book, used here to take advantage of the fact
		// that Python can use negative indices in lists.
		for l in 2..(self.shape.len()) {
			let z = &zs[zs.len() - l];
			let sp = self.activation_prime(z.clone());

			let later_layer = &self.layers[self.layers.len() - l + 1];

			assert_eq!(delta.nrows(), later_layer.weights.nrows());
			let neurons_len_of_this_layer = later_layer.weights.ncols();
			assert_eq!(neurons_len_of_this_layer, neurons_len_of_this_layer);
			assert_eq!(z.nrows(), neurons_len_of_this_layer);
			assert_eq!(sp.nrows(), neurons_len_of_this_layer);

			delta = later_layer.weights.t().dot(&delta) * sp;
			assert_eq!(delta.nrows(), neurons_len_of_this_layer);

			reversed_nabla.push(Layer {
				weights: delta.dot(&activations[activations.len() - l - 1].t()),
				biases: delta.clone(),
			});
		}

		reversed_nabla.reverse();
		reversed_nabla
	}

	/// Returns a tuple `(score, cost)`
	/// where `score` is the number of correct answers which the network predicts
	/// and `cost` is the result of the cost function.
	#[cfg(not(feature = "rayon"))]
	fn evaluate_sum_score_cost(&self, test_data: &[(Matrix, Matrix)]) -> (usize, f64) {
		let n = test_data.len() as f64;
		test_data.iter()
			.fold((0, 0.0), |(mut correct, mut cost), (x, y)| {
				let a = self.feedforward(x);
				cost += self.cost(&a, &y) / n;

				let predict = argmax(&a.column(0));
				let answer = argmax(&y.column(0));
				if predict == answer as usize {
					correct += 1;
				}
				(correct, cost)
			})
	}

	#[cfg(feature = "rayon")]
	fn evaluate_sum_score_cost(&self, test_data: &[(Matrix, Matrix)]) -> (usize, f64) {
		use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

		let n = test_data.len() as f64;
		test_data.par_iter()
			.map(|(x, y)| {
				let a = self.feedforward(x);
				let cost = self.cost(&a, &y) / n;

				let predict = argmax(&a.column(0));
				let answer = argmax(&y.column(0));
				let correct: usize = if predict == answer as usize { 1 } else { 0 };
				(correct, cost)
			})
			.reduce(|| (0_usize, 0.0), |sum, r| (sum.0 + r.0, sum.1 + r.1))
	}

	fn evaluate(&self, test_data: &[(Matrix, Matrix)], lambda: f64) -> (usize, f64) {
		let (score, mut cost) = self.evaluate_sum_score_cost(test_data);
		let n = test_data.len() as f64;
		if lambda > 0.0 {
			cost += 0.5 * (lambda/n) * self.layers.iter()
				.map(|layer| norm(&layer.weights).powi(2))
				.sum::<f64>();
		}
		(score, cost)
	}

	fn activation(&self, mut z: Matrix) -> Matrix {
		z.iter_mut().for_each(|z| *z = ActivationFunction::SIGMOID.f(*z));
		z
	}

	fn activation_prime(&self, mut z: Matrix) -> Matrix {
		z.iter_mut().for_each(|z| *z = ActivationFunction::SIGMOID.prime(*z));
		z
	}

	fn cost(&self, a: &Matrix, y: &Matrix) -> f64 {
		self.cost_fn.f(a, y)
	}

	fn delta(&self, z: &Matrix, a: &Matrix, y: &Matrix) -> Matrix {
		self.cost_fn.delta(z, a, y)
	}
}

fn argmax(a: &ArrayView1<f64>) -> usize {
	a.iter().enumerate().fold((0, 0.0), |(i_max, v_max), (i, &v)| {
		if v > v_max {
			(i, v)
		} else {
			(i_max, v_max)
		}
	}).0
}

fn norm(mat: &Matrix) -> f64 {
	mat.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}
