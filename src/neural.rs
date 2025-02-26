use std::ops::{AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign};

use chrono::Local;
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};

use crate::utils::{sigmoid, sigmoid_prime, Vector1D, WrongAnswers};

// (activation, z)
#[derive(Debug)]
struct FeedForward<T>(T, T);
impl<T> FeedForward<T> {
	fn activation(&self) -> &T {
		&self.0
	}
	fn z(&self) -> &T {
		&self.1
	}
}

#[derive(Debug)]
struct Neuron {
	weights: Vector1D,
	bias: f64,
}
impl Neuron {
	fn new(input_layer_size: usize, rng: &mut ThreadRng) -> Self {
		let weights = (0..input_layer_size).map(|_| rng.random_range(-1.0..1.0)).collect();
		let bias = rng.random_range(-1.0..1.0);
		(weights, bias).into()
	}
	fn zero(input_layer_size: usize) -> Self {
		Self {
			weights: Vector1D::zero(input_layer_size),
			bias: 0.0,
		}
	}
	fn feedforward(&self, inputs: &Vector1D) -> FeedForward<f64> {
		let z = inputs.dot(&self.weights) + self.bias;
		FeedForward(sigmoid(z), z)
	}
	fn backpropagate(&self, delta: f64, a: f64, inputs: &Vector1D) -> DeltaNeuron {
		let sp = sigmoid_prime(a);
		let delta = delta * sp;
		let delta_bias = delta;
		let delta_weights = inputs.clone() * delta;
		DeltaNeuron {
			weights: delta_weights,
			bias: delta_bias,
		}
	}
}
impl From<(Vec<f64>, f64)> for Neuron {
	fn from((weights, bias): (Vec<f64>, f64)) -> Self {
		Self {
			weights: weights.into(),
			bias,
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

#[derive(Debug)]
struct Layer {
	neurons: Vec<Neuron>,
}
impl Layer {
	fn new(input_layer_size: usize, layer_size: usize, rng: &mut ThreadRng) -> Self {
		let neurons = (0..layer_size)
			.map(|_| {
				let weights = (0..input_layer_size).map(|_| rng.random_range(-1.0..1.0)).collect();
				let bias = rng.random_range(-1.0..1.0);
				(weights, bias).into()
			})
			.collect();
		Self { neurons }
	}
	fn zero(input_layer_size: usize, layer_size: usize) -> Self {
		let neurons = (0..layer_size).map(|_| Neuron::zero(input_layer_size)).collect();
		Self { neurons }
	}
	fn feedforward(&self, inputs: &Vector1D) -> FeedForward<Vector1D> {
		assert_eq!(inputs.len(), self.input_layer_size());
		let size = self.neurons.len();
		let ffv = FeedForward(Vector1D::zero(size), Vector1D::zero(size));
		self.neurons.iter()
			.enumerate()
			.fold(ffv, |mut ffv, (i, neuron)| {
				let FeedForward(a, z) = neuron.feedforward(inputs);
				ffv.0[i] = a;
				ffv.1[i] = z;
				ffv
			})
	}
	fn backpropagate(&self, delta: &Vector1D, ff: &FeedForward<Vector1D>, inputs: &Vector1D) -> DeltaLayer {
		assert_eq!(self.size(), delta.len());
		assert_eq!(self.size(), ff.activation().len());
		let delta_neurons: Vec<Neuron> = self.neurons.iter()
			.zip(delta.iter().zip(ff.activation().iter()))
			.map(|(neuron, (&delta, &a))| neuron.backpropagate(delta, a, inputs))
			.collect();
		Self { neurons: delta_neurons }
	}
	fn input_layer_size(&self) -> usize {
		self.neurons[0].weights.len()
	}
	fn size(&self) -> usize {
		self.neurons.len()
	}
	fn prev_layer_size(&self) -> usize {
		self.neurons[0].weights.len()
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

#[derive(Debug)]
pub struct Network {
	layers: Vec<Layer>,
}
impl Network {
	pub fn new (sizes: Vec<u32>) -> Self {
		let mut rng = rand::rng();
		let layers = sizes.windows(2)
			.map(|pair| Layer::new(pair[0] as usize, pair[1] as usize, &mut rng))
			.collect();
		Self { layers }
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

			idx = idx + 1;
		}
	}

	fn update_mini_batch(&mut self, mini_batch: &[Vec<Vec<f64>>], eta: f64) {
		let mut sum_nabla: Vec<DeltaLayer> = self.layers.iter()
			.map(|layer| DeltaLayer::zero(layer.input_layer_size(), layer.size()))
			.collect();

		//let n = &sum_nabla[0].neurons[0];
		//println!("Initial SumNabla.neurons[0]: {{weights=[{},...], bias={}}}", n.weights[0], n.bias);

		for mb in mini_batch {
			// Compute gradients of cost function with backpropagation
			assert_eq!(mb.len(), 2);

			let mini_test_data: Vector1D = mb[0].clone().into();
			let mini_test_label: Vector1D = mb[1].clone().into();
			let nabla = self.backpropagate(&mini_test_data, &mini_test_label);

			// Update gradients
			sum_nabla.iter_mut().zip(nabla.iter())
				.for_each(|(sum, n)| *sum += n);

			//let n = &nabla.last().unwrap().neurons[0];
			//println!("Nabla[last].neurons[0]: {{weights=[{},...], bias={}}} -> SumNabla", n.weights[0], n.bias);
		}

		//let n = &sum_nabla.last().unwrap().neurons[0];
		//println!("SumNabla.neurons[0]: {{weights=[{},...], bias={}}}", n.weights[0], n.bias);

		let m = mini_batch.len();
		let adjustment = sum_nabla.into_iter()
			.map(|n| n * eta / m as f64)
			.collect::<Vec<DeltaLayer>>();

		//let n = &adjustment.last().unwrap().neurons[0];
		//println!("Adjustment.neurons[0]: {{weights=[{},...], bias={}}}", n.weights[0], n.bias);

		self.layers.iter_mut().zip(&adjustment)
			.for_each(|(layer, adj)| *layer -= adj);

		//let n = &self.layers.last().unwrap().neurons[0];
		//println!("Neurons[0]: {{weights=[{},...], bias={}}}", n.weights[0], n.bias);
	}

	// Takes and modifies an input image-vector and returns results
	// based on previously defined weights.
	pub fn feedforward(&self, x: &Vector1D) -> Vector1D {
		let feedforward_for_layer = |a: Vector1D, layer: &Layer| {
			let ff = layer.feedforward(&a);
			ff.activation().iter()
				.map(|a| *a)
				.collect::<Vec<f64>>()
				.into()	
		};
		self.layers.iter()
			.fold(x.clone(), feedforward_for_layer)
	}

	fn feedforward_snapshot(&self, x: &Vector1D) -> Vec<FeedForward<Vector1D>> {
		let a: FeedForward<Vector1D> = FeedForward(x.clone(), x.clone());
		self.layers.iter()
			.fold(vec![], |mut acc, layer| {
				let ff = acc.last().unwrap_or(&a);
				let ff = layer.feedforward(ff.activation());
				acc.push(ff);
				acc
			})
	}

	fn backpropagate(&self, x: &Vector1D, y: &Vector1D) -> Vec<DeltaLayer> {
		// Feedforward part (according to template)
		let ffs = self.feedforward_snapshot(x);
		//let ffs_len: Vec<usize> = ffs.as_slice()[1..].iter().map(|ff| ff.activation().len()).collect();
		//println!("Feedforward snapshot: {:?}", ffs_len);

		//let layers_len: Vec<usize> = self.layers.iter().map(|layer| layer.size()).collect();
		//println!("Layers: {:?}", layers_len);

		// Backward pass
		// Compute error/delta vector and backpropagate the error
		let last_ff = ffs.last().unwrap();
		let mut delta = cost_derivative(&last_ff.activation(), y);

		let mut nabla: Vec<DeltaLayer> = Vec::new();
		for ((i, layer), ff) in self.layers.iter().enumerate().zip(ffs.iter()).rev() {
			let inputs = if i == 0 { x } else { ffs[i-1].activation() };
			let g = layer.backpropagate(&delta, ff, inputs);
			delta = inputs.iter().enumerate()
				.map(|(j, _)| g.neurons.iter().map(|n| n.weights[j]).sum())
				.map(|s: f64| s / inputs.len() as f64)
				.collect::<Vec<f64>>()
				.into();
			nabla.push(g);
		}
		nabla.reverse();
		nabla
	}

	fn evaluate<'a>(&self, test_data: &'a Vec<(Vec<f64>, i32)>) -> (i32, WrongAnswers<'a>) {
		// Returns the sum of correctly assigned test inputs.
		let test_input_size = test_data[0].0.len();
		let mut x_vector = Vector1D::zero(test_input_size);

		let mut wrongs = WrongAnswers::new();
		let r = test_data.iter()
			.map(|(x, y)| {
				x_vector.copy_from_slice(x);

				let r = self.feedforward(&x_vector).find_max().1 as i32;
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

fn cost_derivative(output_activations: &Vector1D, y: &Vector1D) -> Vector1D {
	output_activations.sub(y)
}
