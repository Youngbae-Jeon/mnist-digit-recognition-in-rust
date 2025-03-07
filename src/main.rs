mod network;
mod network2;
mod matrix;
mod utils;
mod io;

use io::TrainData;
use utils::TrainingOptions;

fn main() {
	let data = io::import_images(50_000, 10_000);

	run_network2(data);

	println!("Training successful!");
}

#[allow(dead_code)]
fn run_network1(data: TrainData) {
	use network::Network;

	let mut network = Network::new(&[784, 30, 10]);
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			mini_batch_size: 10,
			eta: 3.0,
			..Default::default()
		},
		&data.training_data,
		&data.test_data,
	);
}

#[allow(dead_code)]
fn run_network2(data: TrainData) {
	use network2::{Network, NetworkOptions};

	let mut network = Network::new(&NetworkOptions {
		shape: vec![784, 30, 10],
		//weights_initializer: utils::WeightInitializer::LARGE,
		//cost: utils::CostFunction::QUADRATIC,
		..Default::default()
	});
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			mini_batch_size: 10,
			eta: 0.5,
			lambda: 5.0,
			..Default::default()
		},
		&data.training_data,
		&data.test_data,
	);
}
