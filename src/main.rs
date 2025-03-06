mod io;
mod neural;
mod neural2;
mod utils;
mod matrix;

use utils::TrainingOptions;

fn main() {
	let data = io::import_images(50_000, 10_000);

	let mut network = neural2::Network::new(&[784, 30, 10]);
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			success_percentage: 0.98,
			mini_batch_size: 10,
			eta: 3.0,
			lambda: 0.0,
		},
		&data.training_data,
		&data.test_data
	);

	println!("Training successful!");
}
