mod network;
mod network2;
mod matrix;
mod utils;
mod io;

use utils::TrainingOptions;

fn main() {
	let data = io::import_images(50_000, 10_000);

	let mut network = network2::Network::new(&[784, 30, 10]);
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			mini_batch_size: 10,
			eta: 0.5,
			lambda: 5.0,
			..Default::default()
		},
		&data.training_data,
		&data.test_data
	);

	println!("Training successful!");
}
