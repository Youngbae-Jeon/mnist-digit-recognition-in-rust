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
use rand::seq::SliceRandom;

use crate::{io::vectorise_num, matrix::Matrix, utils::{ActivationFunction, CostFunction, TrainingOptions, VecInitializer, WeightInitializer}};

struct Layer {
	/// matrix of shape (neurons_len, weights_len)
	weights: Matrix,
	/// matrix of shape (neurons_len, 1)
	biases: Matrix,
}
impl Layer {
	fn size(&self) -> usize {
		self.biases.shape.0
	}
}

pub struct Network {
	shape: Vec<usize>,
	layers: Vec<Layer>,
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
	pub fn new (shape: &[usize]) -> Self {
		Self {
			shape: shape.to_vec(),
			layers: shape.windows(2)
				.map(|w| {
					let weights_len = w[0];
					let neurons_len = w[1];
					let weights_data = WeightInitializer::DEFAULT.f(weights_len * neurons_len);
					let biases_data: Vec<f64> = VecInitializer::STANDARD_NORMAL.f(neurons_len);
					Layer {
						weights: Matrix::new((neurons_len, weights_len), weights_data),
						biases: Matrix::new((neurons_len, 1), biases_data),
					}
				})
				.collect()
		}
	}

	/// Return the output of the network if ``x`` is input.
	fn feedforward(&self, x: &Matrix) -> Matrix {
		let a = self.layers.iter()
			.fold(x.clone(), |x, layer| {
				assert_eq!(x.shape.0, layer.weights.shape.1);
				let z = layer.weights.dot(&x) + &layer.biases;
				let a = self.activation(z);
				assert_eq!(a.shape.0, layer.size());
				assert_eq!(a.shape.1, 1);
				a
			});
		assert_eq!(a.shape, (10, 1));
		a
	}

	/// Train the neural network using mini-batch stochastic gradient
	/// descent.  The ``training_data`` is a list of tuples ``(x, y)``
	/// representing the training inputs and the desired outputs.  The
	/// other non-optional parameters are self-explanatory, as is the
	/// regularization parameter ``lmbda``.  The method also accepts
	/// ``evaluation_data``, usually either the validation or test
	/// data.
	pub fn sgd(
		&mut self,
		options: &TrainingOptions,
		training_data: &[Vec<Vec<f64>>],
		evaluation_data: &[(Vec<f64>, i32)],
	) {
		// change data format to tuples of (Matrix, Matrix) representing (x, y)
		let mut training_data: Vec<(Matrix, Matrix)> = training_data.iter()
			.map(|arr| {
				assert_eq!(arr.len(), 2);
				assert_eq!(arr[0].len(), 784);
				assert_eq!(arr[1].len(), 10);
				let x = Matrix::from(arr[0].clone());
				let y = Matrix::from(arr[1].clone());
				(x, y)
			})
			.collect();
		let evaluation_data: Vec<(Matrix, Matrix)> = evaluation_data.iter()
			.map(|(x, y)| {
				assert_eq!(x.len(), 784);
				let x = Matrix::from(x.to_vec());
				let y = Matrix::from(vectorise_num(&(*y as u8)));
				(x, y)
			})
			.collect();

		println!("Traning with {} data...", training_data.len());
		let (score, cost) = self.evaluate(&evaluation_data, options.lambda);
		let accuracy = score as f64 / evaluation_data.len() as f64;
		println!(" Epoch |  Train   | Train  |   Test   |  Test  | Elapsed ");
		println!("       | Accuracy |  Cost  | Accuracy |  Cost  |   Time  ");
		println!("-------|----------|--------|----------|--------|---------");
		println!(" {:>5} | {:>8} | {:>6} | {:>7.2}% | {:>6.3} | {:>7}", 0, "-", "-", accuracy * 100.0, cost, "-");

		let mut rng = rand::rng();
		for epoch in 0..options.epochs {
			let t = Local::now();
			training_data.shuffle(&mut rng);

			for mini_batch in training_data.chunks(options.mini_batch_size) {
				self.update_mini_batch(mini_batch, options.eta, options.lambda, training_data.len());
			}

			let (train_score, train_cost) = self.evaluate(&training_data, options.lambda);
			let train_accu = train_score as f64 / training_data.len() as f64;

			let (test_score, test_cost) = self.evaluate(&evaluation_data, options.lambda);
			let test_accu = test_score as f64 / evaluation_data.len() as f64;
			let elapsed = (Local::now() - t).num_milliseconds() as f64 / 1000.0;
			println!(" {:>5} | {:>7.2}% | {:>6.3} | {:>7.2}% | {:>6.3} | {:>6.1}s",
				epoch+1, train_accu*100.0, train_cost, test_accu*100.0, test_cost, elapsed);

			if options.success_percentage > 0.0 && test_accu > options.success_percentage {
				println!("Success percentage reached.");
				break;
			}
		}
	}

	/// Update the network's weights and biases by applying gradient
	/// descent using backpropagation to a single mini batch.  The
	/// ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
	/// learning rate, ``lambda`` is the regularization parameter, and
	/// ``n`` is the total size of the training data set.
	pub fn update_mini_batch(&mut self, mini_batch: &[(Matrix, Matrix)], eta: f64, lambda: f64, n: usize) {
		let mut nabla: Vec<Layer> = self.layers.iter()
			.map(|layer| Layer {
				weights: Matrix::zero(layer.weights.shape),
				biases: Matrix::zero(layer.biases.shape),
			})
			.collect();

		for (x, y) in mini_batch {
			//assert_eq!(mb.len(), 2);

			//let x = Matrix::from(mb[0].to_vec());
			//let y = Matrix::from(mb[1].to_vec());

			let delta_nabla = self.backprop(x, y);
			assert_eq!(delta_nabla.len(), nabla.len());

			nabla.iter_mut().zip(delta_nabla.into_iter())
				.for_each(|(n, dn)| {
					n.weights += dn.weights;
					n.biases += dn.biases;
				});
		}

		let m = mini_batch.len() as f64;
		let n = n as f64;
		self.layers.iter_mut().zip(nabla)
			.for_each(|(layer, nabla)| {
				layer.weights *= 1.0 - eta * (lambda / n); // L2 regularization
				layer.weights -= nabla.weights * (eta / m);
				layer.biases -= nabla.biases * (eta / m);
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
			//weights: delta.dot(&activations[activations.len() - 2].transpose()),
			weights: delta.dot_transpose(&activations[activations.len() - 2]),
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

			assert_eq!(delta.shape.0, later_layer.weights.shape.0);
			let neurons_len_of_this_layer = later_layer.weights.shape.1;
			assert_eq!(neurons_len_of_this_layer, neurons_len_of_this_layer);
			assert_eq!(z.shape.0, neurons_len_of_this_layer);
			assert_eq!(sp.shape.0, neurons_len_of_this_layer);

			delta = later_layer.weights.transpose_dot(&delta) * sp;
			assert_eq!(delta.shape.0, neurons_len_of_this_layer);

			reversed_nabla.push(Layer {
				weights: delta.dot_transpose(&activations[activations.len() - l - 1]),
				biases: delta.clone(),
			});
		}

		reversed_nabla.reverse();
		reversed_nabla
	}

	/// Return the number of test inputs for which the neural
	/// network outputs the correct result. Note that the neural
	/// network's output is assumed to be the index of whichever
	/// neuron in the final layer has the highest activation.
	fn evaluate(&self, test_data: &[(Matrix, Matrix)], lambda: f64) -> (usize, f64) {
		let n = test_data.len() as f64;
		let (correct, mut cost) = test_data.iter()
			.fold((0, 0.0), |(mut correct, mut cost), (x, y)| {
				//let x = Matrix::from(test_data[0].clone());
				//let y = Matrix::from(test_data[1].clone());
				assert_eq!(x.shape, (784, 1));
				assert_eq!(y.shape, (10, 1));
				let a = self.feedforward(x);
				cost += self.cost(&a, &y) / n;

				let yhat = a.column(0).argmax();
				let y = y.column(0).argmax();
				if yhat == y as usize {
					correct += 1;
				}
				
				(correct, cost)
			});
		cost += 0.5 * (lambda/n) * self.layers.iter()
			.map(|layer| norm(&layer.weights).powi(2))
			.sum::<f64>();
		(correct, cost)
	}

	fn activation(&self, z: Matrix) -> Matrix {
		Matrix::update(z, |z| ActivationFunction::SIGMOID.f(z))
	}

	fn activation_prime(&self, z: Matrix) -> Matrix {
		Matrix::update(z, |z| ActivationFunction::SIGMOID.prime(z))
	}

	fn cost(&self, a: &Matrix, y: &Matrix) -> f64 {
		CostFunction::CROSS_ENTROPY.f(a, y)
	}

	fn delta(&self, z: &Matrix, a: &Matrix, y: &Matrix) -> Matrix {
		CostFunction::CROSS_ENTROPY.delta(z, a, y)
	}
}

fn norm(x: &Matrix) -> f64 {
	x.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}
