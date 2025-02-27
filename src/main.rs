mod network;
mod io;
mod neural;
mod utils;
use std::env;

use network::Network;
use utils::sigmoid;

fn main() {
	get_runner()();
}

fn get_runner() -> fn() {
	let arg = env::args().nth(1);
	match arg {
		Some(arg) => {
			match arg.as_str() {
				"1" => run1,
				"2" => run2,
				_ => panic!("Invalid argument!"),
			}
		},
		None => panic!("No argument provided!"),
	}
}

const SUCCESS_RATE: f64 = 0.95;

fn run1() {
	let data = io::import_images();

	let layers = vec![784, 16, 10];

	let mut net = network::Network::new(layers.clone());
	train_network(&mut net, &data);

	println!("Initialised network with layer structure {:?}", layers);

	test_input_images(&net);
}

fn test_input_images(net: &Network) {
	let input = io::TestImages::new("data/test2.png");

	let guess = network::argmax(&net.feedforward(&input.pixel_vector2d));

	println!("Pixel structure is: {:?}", input.pixel_vector2d);

	println!("The network guesses the number is: {}", guess)
}

fn train_network(net: &mut network::Network, data: &io::TrainData) {
	let mbs: usize = 10;
	let learning_rate: f64 = 0.25;

	net.sgd(
		&data.training_data,
		200, 
		SUCCESS_RATE,
		mbs, 
		learning_rate, 
		&data.test_data
	);

	println!("Training successful!")
}

fn run2() {
	let data = io::import_images();

	let layers = vec![784, 16, 10];

	let mut net = neural::Network::new(layers.clone());
	train_neural(&mut net, &data)
}

fn learning_rate_fn(epoch: usize, accuracy: f64) -> f64 {
	((sigmoid(1.0 - accuracy) - 0.5) * 15.0).min(1.0)
}

fn train_neural(net: &mut neural::Network, data: &io::TrainData) {
	let mbs: usize = 10;

	net.sgd(
		&data.training_data,
		200, 
		SUCCESS_RATE,
		mbs, 
		learning_rate_fn,
		&data.test_data
	);

	println!("Training successful!")
}
