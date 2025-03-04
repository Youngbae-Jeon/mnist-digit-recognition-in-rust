use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign};

use chrono::Local;
use rand::{seq::SliceRandom, Rng};

use crate::utils::{ActivationFunction, CostFunction, Vector1D, WeightsInitializer, WrongAnswers};

#[derive(Debug)]
struct Neuron {
	weights: Vector1D,
	bias: f64,
}
impl Neuron {
	fn new(weights_len: usize, wi: WeightsInitializer) -> Self {
		let mut rng = rand::rng();
		Self {
			weights: (wi.f)(weights_len),
			bias: rng.random_range(-1.0..1.0),
		}
	}
	fn zero(weights_len: usize) -> Self {
		Self {
			weights: Vector1D::zero(weights_len),
			bias: 0.0,
		}
	}
	fn feedforward(&self, inputs: &Vector1D) -> f64 {
		inputs.dot(&self.weights) + self.bias
	}
	// returns gradients of weights and bias
	fn backpropagate(&self, delta: f64, inputs: &Vector1D) -> DeltaNeuron {
		DeltaNeuron {
			weights: inputs.clone() * delta,
			bias: delta,
		}
	}
}
impl AddAssign<&Self> for Neuron {
	fn add_assign(&mut self, other: &Self) {
		self.weights += &other.weights;
		self.bias += other.bias;
	}
}
impl SubAssign<&Self> for Neuron {
	fn sub_assign(&mut self, other: &Self) {
		self.weights -= &other.weights;
		self.bias -= other.bias;
	}
}
impl MulAssign<f64> for Neuron {
	fn mul_assign(&mut self, rhs: f64) {
		self.weights.iter_mut().for_each(|w| *w *= rhs);
		self.bias *= rhs;
	}
}
impl Mul<f64> for Neuron {
	type Output = Neuron;

	fn mul(mut self, rhs: f64) -> Self::Output {
		self *= rhs;
		self
	}
}
impl DivAssign<f64> for Neuron {
	fn div_assign(&mut self, rhs: f64) {
		self.weights /= rhs;
		self.bias /= rhs;
	}
}

type DeltaNeuron = Neuron;

struct Layer {
	neurons: Vec<Neuron>,
	af: ActivationFunction,
}
impl Layer {
	fn new(input_layer_size: usize, layer_size: usize, wi: WeightsInitializer, af: ActivationFunction) -> Self {
		Self {
			neurons: (0..layer_size)
				.map(|_| Neuron::new(input_layer_size, wi))
				.collect(),
			af,
		}
	}
	fn zero(input_layer_size: usize, layer_size: usize, af: ActivationFunction) -> Self {
		Self {
			neurons: (0..layer_size)
				.map(|_| Neuron::zero(input_layer_size))
				.collect(),
			af,
		}
	}
	/// Returns a tuple of (activations, z)
	fn feedforward(&self, inputs: &Vector1D) -> (Vector1D, Vector1D) {
		assert_eq!(inputs.len(), self.input_layer_size());
		let size = self.size();
		let az = (Vector1D::zero(size), Vector1D::zero(size));
		self.neurons.iter().enumerate()
			.fold(az, |mut az, (i, neuron)| {
				let z = neuron.feedforward(inputs);
				az.0[i] = self.apply_activation(z);
				az.1[i] = z;
				az
			})
	}
	fn backpropagate(&self, delta: &Vector1D, inputs: &Vector1D) -> DeltaLayer {
		assert_eq!(self.size(), delta.len());
		let delta_neurons: Vec<Neuron> = self.neurons.iter()
			.zip(delta.iter())
			.map(|(neuron, &delta)| neuron.backpropagate(delta, inputs))
			.collect();
		Self { neurons: delta_neurons, af: self.af }
	}
	fn input_layer_size(&self) -> usize {
		self.neurons[0].weights.len()
	}
	fn size(&self) -> usize {
		self.neurons.len()
	}
	fn apply_activation(&self, z: f64) -> f64 {
		(self.af.f)(z)
	}
	fn activation_prime_of(&self, z: f64) -> f64 {
		(self.af.prime)(z)
	}
	fn iter_nth_weights_of_neurons(&self, n: usize) -> impl Iterator<Item = f64> + '_ {
		self.neurons.iter().map(move |neuron| neuron.weights[n])
	}
	fn iter_all_weights(&self) -> impl Iterator<Item = f64> + '_ {
		self.neurons.iter().flat_map(|neuron| neuron.weights.iter().map(|&w| w))
	}
	fn iter_mut_all_weights(&mut self) -> impl Iterator<Item = &mut f64> {
		self.neurons.iter_mut().flat_map(|neuron| neuron.weights.iter_mut())
	}
}
impl AddAssign<&Self> for Layer {
	fn add_assign(&mut self, rhs: &Self) {
		assert_eq!(self.size(), rhs.size());
		self.neurons.iter_mut()
			.zip(rhs.neurons.iter())
			.for_each(|(n, on)| *n += on);
	}
}
impl SubAssign<&DeltaLayer> for Layer {
	fn sub_assign(&mut self, rhs: &DeltaLayer) {
		self.neurons.iter_mut()
			.zip(rhs.neurons.iter())
			.for_each(|(n, on)| *n -= on);
	}
}
impl MulAssign<f64> for Layer {
	fn mul_assign(&mut self, rhs: f64) {
		self.neurons.iter_mut().for_each(|n| *n *= rhs);
	}
}
impl Mul<f64> for Layer {
	type Output = Layer;

	fn mul(mut self, rhs: f64) -> Self::Output {
		self.neurons.iter_mut().for_each(|n| *n *= rhs);
		self
	}
}
impl DivAssign<f64> for Layer {
	fn div_assign(&mut self, rhs: f64) {
		self.neurons.iter_mut().for_each(|n| *n /= rhs);
	}
}
impl Div<f64> for Layer {
	type Output = Layer;

	fn div(mut self, rhs: f64) -> Self::Output {
		self.neurons.iter_mut().for_each(|n| *n /= rhs);
		self
	}
}

type DeltaLayer = Layer;

pub struct Network {
	layers: Vec<Layer>,
	cf: CostFunction,
}
impl Network {
	pub fn new (sizes: &[usize]) -> Self {
		let layers = sizes.windows(2)
			.map(|pair| {
				let wi = WeightsInitializer::DEFAULT;
				let af = ActivationFunction::SIGMOID;
				Layer::new(pair[0] as usize, pair[1] as usize, wi, af)
			})
			.collect();
		Self {
			layers,
			cf: CostFunction::CROSS_ENTROPY,
		}
	}

	pub fn sgd(
		&mut self,
		options: &TrainingOptions,
		training_data: &Vec<Vec<Vec<f64>>>,
		test_data: &Vec<(Vec<f64>, i32)>,
	) {
		let TrainingOptions { mini_batch_size, eta, epochs, success_percentage, lambda } = *options;

		// Print evaluation result
		let (correct_images, _, _) = self.evaluate(test_data);
		let mut accuracy = correct_images as f64 / test_data.len() as f64;
		println!("Epoch[0] {:.1}% ({}/{})", accuracy * 100.0, correct_images, test_data.len());

		let mut td = training_data.clone();
		let mut rng = rand::rng();

		let mut idx = 0;
		while idx < epochs && accuracy < success_percentage {
			let t = Local::now();

			// learn from shuffled training data
			td.shuffle(&mut rng);
			let cost = self.start_learning(&td, eta, mini_batch_size, lambda);

			// Print evaluation result
			let (correct_images, eval_cost, _wrong_answers) = self.evaluate(test_data);
			accuracy = correct_images as f64 / test_data.len() as f64;

			//_wrong_answers.dump();
			println!("Epoch[{}] {:.1}% ({}/{}, eta={:.1}%, cost(t)={:.4}, cost(e)={:.4}, {}s delayed)",
				idx+1, accuracy * 100.0, correct_images, test_data.len(),
				eta * 100.0, cost, eval_cost,
				(Local::now() - t).num_seconds());

			idx = idx + 1;
		}
	}

	// returns the average cost
	fn start_learning(&mut self, traning_data: &[Vec<Vec<f64>>], eta: f64, mini_batch_size: usize, lambda: f64) -> f64 {
		let mini_batches = traning_data.chunks(mini_batch_size);
		let mut sum_cost = 0.0;

		// Apply backpropagation on mini_batches
		for mb in mini_batches {
			sum_cost += self.update_mini_batch(mb, eta, lambda, traning_data.len());
		}
		sum_cost
	}

	// returns the cost sum of the mini_batch
	fn update_mini_batch(&mut self, mini_batch: &[Vec<Vec<f64>>], eta: f64, lambda: f64, train_data_len: usize) -> f64 {
		let mut gradients: Vec<DeltaLayer> = self.layers.iter()
			.map(|layer| DeltaLayer::zero(layer.input_layer_size(), layer.size(), layer.af))
			.collect();
		let mut sum_cost = 0.0;

		for mb in mini_batch {
			assert_eq!(mb.len(), 2);

			let data: Vector1D = mb[0].clone().into();
			let label: Vector1D = mb[1].clone().into();
			let (g, cost) = self.backpropagate(&data, &label);

			// variable `gradients` stands for sum of gradients in mini_batch
			gradients.iter_mut().zip(g.iter())
				.for_each(|(sum, n)| *sum += n);
			sum_cost += cost / train_data_len as f64;
		}

		// Regularization
		if lambda != 0.0 {
			let sum_of_squares_of_all_weights: f64 = self.iter_all_weights()
					.map(|w| w.powi(2))
					.sum::<f64>();
			assert!(!sum_of_squares_of_all_weights.is_nan());
			sum_cost += (lambda / (2.0 * train_data_len as f64) * sum_of_squares_of_all_weights) / train_data_len as f64;
		}

		// Update gradients
		let m = mini_batch.len();
		self.layers.iter_mut().zip(gradients.into_iter())
			.for_each(|(layer, gradients_for_layer)| {
				if lambda != 0.0 {
					let weight_decay_factor = 1.0 - eta * lambda / train_data_len as f64;
					let compute_delta_applied_weight = |w: f64, dw: f64| -> f64 {
						weight_decay_factor * w - (eta / m as f64) * dw
					};
					layer.neurons.iter_mut().zip(gradients_for_layer.neurons.iter())
						.for_each(|(neuron, delta_neuron)| {
							neuron.weights.iter_mut().zip(delta_neuron.weights.iter())
								.for_each(|(w, dw)| *w = compute_delta_applied_weight(*w, *dw));
							neuron.bias -= (eta / m as f64) * delta_neuron.bias;
						});
				} else {
					*layer -= &(gradients_for_layer * (eta / m as f64));
				}
			});

		sum_cost
	}

	// Takes and modifies an input image-vector and returns results
	// based on previously defined weights.
	pub fn feedforward(&self, x: &Vector1D) -> Vector1D {
		self.layers.iter()
			.fold(x.clone(), |a, layer| {
				layer.feedforward(&a).0.iter()
					.map(|a| *a)
					.collect()
			})
	}

	pub fn predict(&self, x: &Vector1D) -> i32 {
		self.feedforward(x).find_max().1 as i32
	}

	fn feedforward_snapshot(&self, x: &Vector1D) -> Vec<(Vector1D, Vector1D)> {
		let a = (x.clone(), x.clone());
		self.layers.iter()
			.fold(Vec::with_capacity(self.layers.len()), |mut azs, layer| {
				let inputs = &azs.last().unwrap_or(&a).0;
				let az = layer.feedforward(inputs);
				azs.push(az);
				azs
			})
	}

	// returns (gradients of weights and biases, cost)
	fn backpropagate(&self, x: &Vector1D, y: &Vector1D) -> (Vec<DeltaLayer>, f64) {
		let azs = self.feedforward_snapshot(x);

		let last_a = &azs.last().unwrap().0;
		let cost = (self.cf.f)(last_a, y);
		let mut delta: Vector1D = (self.cf.derivative)(last_a, y);

		let mut nabla: Vec<DeltaLayer> = Vec::with_capacity(self.layers.len());
		for (i, layer) in self.layers.iter().enumerate().rev() {
			let (_, z) = &azs[i];
			let inputs = if i == 0 { x } else { &azs[i-1].0 };
			let sp: Vector1D = z.iter().map(|&z| layer.activation_prime_of(z)).collect();

			if i == self.layers.len() - 1 {
				delta *= &sp;
			} else {
				let next_layer = &self.layers[i+1];
				assert_eq!(next_layer.input_layer_size(), sp.len());
				delta = sp.iter().enumerate()
					.map(|(i, sp)| {
						let d: f64 = next_layer.iter_nth_weights_of_neurons(i)
							.zip(delta.iter())
							.map(|(w, d)| w * d)
							.sum();
						d * sp
					})
					.collect();
			};

			let g = layer.backpropagate(&delta, inputs);
			nabla.push(g);
		}
		
		nabla.reverse();
		(nabla, cost)
	}

	// returns (the number of correct answers, cost, and wrong answers)
	fn evaluate<'a>(&self, test_data: &'a Vec<(Vec<f64>, i32)>) -> (i32, f64, WrongAnswers<'a>) {
		// Returns the sum of correctly assigned test inputs.
		let test_input_size = test_data[0].0.len();
		let mut x_vector = Vector1D::zero(test_input_size);
		let mut y_vector = Vector1D::zero(10);

		let mut wrongs = WrongAnswers::new();
		let mut cost = 0.0;
		let r = test_data.iter()
			.map(|(x, digit)| {
				x_vector.copy_from_slice(x);
				y_vector.fill(0.0);
				y_vector[*digit as usize] = 1.0;

				let a = self.feedforward(&x_vector);
				cost += (self.cf.f)(&a, &y_vector) / test_data.len() as f64;

				let (_possibility, guess) = a.find_max();
				if guess as i32 == *digit {
					1
				} else {
					wrongs.push((x, *digit, guess as i32));
					0
				}
			})
			.sum();
		(r, cost, wrongs)
	}

	fn iter_all_weights(&self) -> impl Iterator<Item = f64> + '_ {
		self.layers.iter().flat_map(|layer| layer.iter_all_weights())
	}
}

pub struct TrainingOptions {
	pub epochs: usize,
	pub success_percentage: f64,
	pub mini_batch_size: usize,
	pub eta: f64,
	pub lambda: f64,
}
