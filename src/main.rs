mod network;
mod io;
mod neural;
mod utils;
use network::Network;

fn main() {
	run2()
}

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
	let learning_rate: f64 = 0.5;
	let success: f64 = 0.9;

	net.sgd(
		&data.training_data,
		200, 
		success,
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

fn train_neural(net: &mut neural::Network, data: &io::TrainData) {
	let mbs: usize = 10;
	let learning_rate: f64 = 0.5;
	let success: f64 = 0.9;

	net.sgd(
		&data.training_data,
		200, 
		success,
		mbs, 
		learning_rate, 
		&data.test_data
	);

	println!("Training successful!")
}
