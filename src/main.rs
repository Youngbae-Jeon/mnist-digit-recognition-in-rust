mod io;
mod neural;
mod utils;

use neural::TrainingOptions;

fn main() {
	let data = io::import_images();

	let mut network = neural::Network::new(&[784, 30, 10]);
	network.sgd(
		&TrainingOptions {
			epochs: 30,
			success_percentage: 0.98,
			mini_batch_size: 10,
			eta: 0.1,
		},
		&data.training_data,
		&data.test_data
	);

	println!("Training successful!");
}
