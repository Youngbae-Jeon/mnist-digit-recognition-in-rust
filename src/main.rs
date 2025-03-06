mod io;
mod network;
mod utils;
mod matrix;

use utils::TrainingOptions;

fn main() {
	let data = io::import_images(50_000, 10_000);

	let mut network = network::Network::new(&[784, 30, 10]);
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			mini_batch_size: 10,
			eta: 3.0,
			..Default::default()
		},
		&data.training_data,
		&data.test_data
	);

	println!("Training successful!");
}
