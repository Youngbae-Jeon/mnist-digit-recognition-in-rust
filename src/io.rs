extern crate mnist;
use mnist::{Mnist, MnistBuilder};

pub struct TrainData {
	//train_array_2d: Vec<Vec<f32>>,
	//train_label: Vec<u8>,
	pub training_data: Vec<Vec<Vec<f32>>>,
	//validation_array_2d: Vec<Vec<f32>>,
	//validation_label: Vec<u8>,
	pub validation_data: Vec<(Vec<f32>, i32)>,
	//test_array_2d: Vec<Vec<f32>>,
	//test_label: Vec<u8>,
	pub test_data: Vec<(Vec<f32>, i32)>,
}

pub fn import_images(training_size: usize, test_size: usize) -> TrainData {
	// Code was partly taken from this address (https://docs.rs/mnist/0.4.1/mnist/)
	let (rows, cols) = (28, 28);

	// Deconstruct the returned Mnist struct.
	let Mnist { trn_img, trn_lbl, 
				val_img, val_lbl, 
				tst_img, tst_lbl } = MnistBuilder::new()
		.label_format_digit()
		.training_set_length(training_size as u32)
		.validation_set_length(test_size as u32)
		.test_set_length(test_size as u32)
		.base_path("data/")
		.finalize();
	
	let trn_array2d =  pack_images_vec(trn_img, rows, cols);
	let val_array2d = pack_images_vec(val_img, rows, cols);
	let tst_array2d = pack_images_vec(tst_img, rows, cols);
	
	TrainData {
		training_data: pack_training_data(&trn_array2d, &trn_lbl),
		validation_data: pack_vector_data(&val_array2d, &val_lbl),
		test_data: pack_vector_data(&tst_array2d, &tst_lbl),
		//train_array_2d: trn_array2d,
		//train_label: trn_lbl,
		//validation_array_2d: val_array2d,
		//validation_label: val_lbl,
		//test_array_2d: tst_array2d,
		//test_label: tst_lbl,
	}
}

// Designed to normalize Pixels without color-data so 255 is hardcoded
fn normalize(x: &u8) -> f32 {
	*x as f32 / 255.0
}

fn pack_images_vec(input_vector: Vec<u8>, rows: u8, cols: u8) -> Vec<Vec<f32>> {
	let input: Vec<f32> = input_vector
		.iter()
		.map(normalize)
		.collect();
	
	let image_len = (rows as i32 * cols as i32) as usize;
	let target_vec: Vec<Vec<f32>> = input.chunks(image_len)
		.map(|x| x.to_vec())    
		.collect();
	
	target_vec
}

fn pack_training_data(data: &[Vec<f32>], label: &[u8]) -> Vec<Vec<Vec<f32>>> {
	let mut td: Vec<Vec<Vec<f32>>> = Vec::new();
	let num_vec: Vec<Vec<f32>> = label.iter().map(vectorise_num).collect();

	for (x, y) in data.iter().zip(&num_vec) {
		td.push(vec![x.clone(), y.clone()])
	}
	td
}

fn pack_vector_data(data: &[Vec<f32>], label: &Vec<u8>) -> Vec<(Vec<f32>, i32)>{
	let mut result: Vec<(Vec<f32>, i32)> = Vec::new();
	for (x, y) in data.iter().zip(label) {
		result.push((x.clone(), *y as i32))
	}
	result
}

// Creates a 1d vector with size 10 (0..9) which contains a 1 a the 
// index-location of the input @num
pub fn vectorise_num(num: &u8) -> Vec<f32> {
	let mut out = vec![0.; 10];
	out[*num as usize] = 1.;
	out
}