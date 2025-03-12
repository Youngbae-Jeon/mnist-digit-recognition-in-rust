// ported from network.py
// ----------------------
// A module to implement the stochastic gradient descent learning
// algorithm for a feedforward neural network.  Gradients are calculated
// using backpropagation.  Note that I have focused on making the code
// simple, easily readable, and easily modifiable.  It is not optimized,
// and omits many desirable features.

use chrono::Local;
use rand::{seq::SliceRandom, Rng};
use rand_distr::StandardNormal;

use crate::{matrix::Matrix, utils::{sigmoid, sigmoid_prime, TrainingOptions}};

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
	/// The list ``shape`` contains the number of neurons in the
	/// respective layers of the network.  For example, if the list
	/// was [2, 3, 1] then it would be a three-layer network, with the
	/// first layer containing 2 neurons, the second layer 3 neurons,
	/// and the third layer 1 neuron.  The biases and weights for the
	/// network are initialized randomly, using a Gaussian
	/// distribution with mean 0, and variance 1.  Note that the first
	/// layer is assumed to be an input layer, and by convention we
	/// won't set any biases for those neurons, since biases are only
	/// ever used in computing the outputs from later layers.
	pub fn new (shape: &[usize]) -> Self {
		Self {
			shape: shape.to_vec(),
			layers: shape.windows(2)
				.map(|w| {
					let weights_len = w[0];
					let neurons_len = w[1];
					let weights_data: Vec<f64> = (0..(weights_len * neurons_len))
						.map(|_| rand::rng().sample(StandardNormal))
						// .map(|_| rng.random_range(-1.0..1.0))
						.collect();
					let biases_data: Vec<f64> = (0..neurons_len)
						.map(|_| rand::rng().sample(StandardNormal))
						// .map(|_| rng.random_range(-1.0..1.0))
						.collect();
					Layer {
						weights: Matrix::new((neurons_len, weights_len), weights_data),
						biases: Matrix::new((neurons_len, 1), biases_data),
					}
				})
				.collect()
		}
	}

	/// Return the output of the network if ``x`` is input.
	fn feedforward(&self, x: Matrix) -> Matrix {
		let ff = self.layers.iter()
			.fold(x, |x, layer| {
				assert_eq!(x.shape.0, layer.weights.shape.1);
				let z = layer.weights.dot(&x) + &layer.biases;
				let a = Matrix::update(z, sigmoid);
				assert_eq!(a.shape.0, layer.size());
				assert_eq!(a.shape.1, 1);
				a
			});
		assert_eq!(ff.shape, (10, 1));
		ff
	}

	/// Train the neural network using mini-batch stochastic
	/// gradient descent.  The ``training_data`` is a list of tuples
	/// ``(x, y)`` representing the training inputs and the desired
	/// outputs.  The other non-optional parameters are
	/// self-explanatory.  If ``test_data`` is provided then the
	/// network will be evaluated against the test data after each
	/// epoch, and partial progress printed out.  This is useful for
	/// tracking progress, but slows things down substantially.
	pub fn sgd(
		&mut self,
		options: &TrainingOptions,
		training_data: &[Vec<Vec<f64>>],
		test_data: &[(Vec<f64>, i32)],
	) {
		let score = self.evaluate(test_data);
		let accuracy = score as f64 / test_data.len() as f64;
		println!("Epoch.0: {:.2}% successful / {} tests (Before learning)", accuracy * 100.0, test_data.len());

		let mut rng = rand::rng();
		for epoch in 0..options.epochs {
			let t = Local::now();
			let mut training_data = training_data.to_owned();
			training_data.shuffle(&mut rng);

			for mini_batch in training_data.chunks(options.mini_batch_size) {
				self.update_mini_batch(mini_batch, options.eta);
			}

			let score = self.evaluate(test_data);
			let accuracy = score as f64 / test_data.len() as f64;
			let elapsed = (Local::now() - t).num_milliseconds() as f64 / 1000.0;
			println!("Epoch.{}: {:.2}% successful / {} tests ({:.1}s elapsed)", epoch + 1, accuracy * 100.0, test_data.len(), elapsed);
		}
	}

	/// Update the network's weights and biases by applying
	/// gradient descent using backpropagation to a single mini batch.
	/// The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
	/// is the learning rate.
	pub fn update_mini_batch(&mut self, mini_batch: &[Vec<Vec<f64>>], eta: f64) {
		let mut nabla: Vec<Layer> = self.layers.iter()
			.map(|layer| Layer {
				weights: Matrix::zero(layer.weights.shape),
				biases: Matrix::zero(layer.biases.shape),
			})
			.collect();

		for mb in mini_batch {
			assert_eq!(mb.len(), 2);

			let x = Matrix::from(mb[0].to_vec());
			let y = Matrix::from(mb[1].to_vec());

			let delta_nabla = self.backprop(x, y);
			assert_eq!(delta_nabla.len(), nabla.len());

			nabla.iter_mut().zip(delta_nabla.into_iter())
				.for_each(|(n, dn)| {
					n.weights += dn.weights;
					n.biases += dn.biases;
				});
		}

		let m = mini_batch.len() as f64;
		self.layers.iter_mut().zip(nabla)
			.for_each(|(layer, n)| {
				layer.weights -= n.weights * (eta / m);
				layer.biases -= n.biases * (eta / m);
			});
	}

	/// Return a struct ``Layer { weights: nabla_w, biases: nabla_b }`` representing the
	/// gradient for the cost function C_x.
	fn backprop(&self, x: Matrix, y: Matrix) -> Vec<Layer> {
		let mut reversed_nabla: Vec<Layer> = Vec::with_capacity(self.layers.len());

		// feedforward
		let mut activations = Vec::with_capacity(self.layers.len() + 1); // list to store all the activations, layer by layer
		activations.push(x);
		let mut zs: Vec<Matrix> = Vec::with_capacity(self.layers.len()); // list to store all the z vectors, layer by layer

		for layer in self.layers.iter() {
			let x = activations.last().unwrap();
			let z = layer.weights.dot(x) + &layer.biases;
			zs.push(z.clone());

			let a = Matrix::update(z, sigmoid);
			activations.push(a);
		}

		// backward pass
		let cost_derivative = activations.last().unwrap().clone() - y;
		let sp = Matrix::update(zs.last().unwrap().clone(), sigmoid_prime);
		assert_eq!(cost_derivative.shape, sp.shape);

		let mut delta = cost_derivative * sp;
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
			let sp = Matrix::update(z.clone(), sigmoid_prime);

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
	fn evaluate(&self, test_data: &[(Vec<f64>, i32)]) -> usize {
		test_data.iter()
			.map(|(x, y)| {
				let x = Matrix::from(x.clone());
				let ff = self.feedforward(x);
				(ff.column(0).argmax(), *y as usize)
			})
			.filter(|(x, y)| x == y)
			.count()
	}
}
