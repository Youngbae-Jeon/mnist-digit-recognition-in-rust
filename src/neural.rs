use chrono::Local;
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use utils::math::sigmoid;

use crate::utils::{sigmoid_prime, Vector1D, WrongAnswers};

// (activation, z)
type FeedForward<T> = (T, T);

struct Neuron {
	weights: Vector1D,
	bias: f64,
}
impl Neuron {
	fn feedforward(&self, a: &Vector1D) -> FeedForward<f64> {
		let z = a.dot(&self.weights) + self.bias;
		(sigmoid(z), z)
	}
	fn backpropagate(&self, delta: f64, z: f64) -> DeltaNeuron {
		let sp = sigmoid_prime(z);
		let delta = delta * sp;
		let delta_bias = delta;
		let delta_weights = self.weights.mul(delta);
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

type DeltaNeuron = Neuron;

struct Layer {
	neurons: Vec<Neuron>,
}
impl Layer {
	fn new (input_layer_size: usize, layer_size: usize, rng: &mut ThreadRng) -> Self {
		let neurons = (0..layer_size)
			.map(|_| {
				let weights = (0..input_layer_size).map(|_| rng.random_range(-1.0..1.0)).collect();
				let bias = rng.random_range(-1.0..1.0);
				(weights, bias).into()
			})
			.collect();
		Self { neurons }
	}
	fn feedforward(&self, a: &Vector1D) -> FeedForward<Vector1D> {
		assert_eq!(a.len(), self.input_layer_size());
		let size = self.neurons.len();
		let ff = (Vector1D::with_capacity(size), Vector1D::with_capacity(size));
		self.neurons.iter()
			.enumerate()
			.fold(ff, |mut ff, (i, neuron)| {
				let (a, z) = neuron.feedforward(a);
				ff.0[i] = a;
				ff.1[i] = z;
				ff
			})
	}
	fn backpropagate(&self, delta: f64, ff: &FeedForward<Vector1D>) -> DeltaLayer {
		assert_eq!(self.neurons.len(), ff.0.len());
		let delta_neurons: Vec<Neuron> = self.neurons.iter()
			.zip(ff.1.iter())
			.map(|(neuron, z)| neuron.backpropagate(delta, *z))
			.collect();
		Self { neurons: delta_neurons }
	}
	fn input_layer_size(&self) -> usize {
		self.neurons[0].weights.len()
	}
}

type DeltaLayer = Layer;

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

			//println!("weights are: {:?}", self.weights[0][0]);
			idx = idx + 1;
		}
	}

	fn update_mini_batch(&mut self, mini_batch: &[Vec<Vec<f64>>], eta: f64) {
		// Used to update weights and biases through a single gradient
		// descent iteration

		// Initialise gradient vectors
		//println!("WEIGHTS 10: {:?}", &self.weights[0][0][..10]);

		//let mut nabla_b: Vec<Vec<f64>> = Vec::new();

		//for bias in &self.biases {
		//	nabla_b.push(vec![0.; bias.len()]);
		//}

		//let mut nabla_w: Vec<Vec<Vec<f64>>> = Vec::new();

		//for weight in &self.weights {
		//	let mut w: Vec<Vec<f64>> = Vec::new();
		//	for ws in weight {
		//		w.push(vec![0.; ws.len()]);
		//	}
		//	nabla_w.push(w)
		//}

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


	// Takes and modifies an input image-vector and returns results
	// based on previously defined weights.
	pub fn feedforward(&self, x: &Vector1D) -> Vector1D {
		let feedforward_for_layer = |a: Vector1D, layer: &Layer| {
			let (activations, _zs) = layer.feedforward(&a);
			activations.iter()
				.map(|a| *a)
				.collect::<Vec<f64>>()
				.into()	
		};
		self.layers.iter()
			.fold(x.clone(), feedforward_for_layer)
	}

	pub fn feedforward_snapshot(&self, x: &Vector1D) -> Vec<FeedForward<Vector1D>> {
		let a: FeedForward<Vector1D> = (x.clone(), x.clone());
		self.layers.iter()
			.fold(vec![a], |mut acc, layer| {
				let a = acc.last().unwrap();
				let ff = layer.feedforward(&a.0);
				acc.push(ff);
				acc
			})
	}

	fn backpropagate(&self, x: &Vector1D, y: &Vector1D) -> Vec<DeltaLayer> {
		// Calculates the gradient for the cost-function
		let mut nabla: Vec<DeltaLayer> = Vec::new();

		// Feedforward part (according to template)
		let ffs = self.feedforward_snapshot(x);

		// Backward pass
		// Compute error/delta vector and backpropagate the error
		let delta_cost = cost_derivative(&ffs.last().unwrap().0, y);

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


	fn evaluate<'a>(&self, test_data: &'a Vec<(Vec<f64>, i32)>) -> (i32, WrongAnswers<'a>) {
		// Returns the sum of correctly assigned test inputs.
		let test_input_size = test_data[0].0.len();
		let mut x_vector = Vector1D::with_capacity(test_input_size);

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
