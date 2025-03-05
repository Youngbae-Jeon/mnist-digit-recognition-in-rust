use std::{fmt::Display, ops::{Add, AddAssign, Index, Mul, MulAssign, Sub, SubAssign}};

#[derive(Clone)]
pub struct Matrix {
	pub shape: (usize, usize),
	data: Vec<f64>,
}
impl Matrix {
	pub fn update(mut mat: Matrix, f: fn(f64) -> f64) -> Matrix {
		mat.iter_mut()
			.for_each(|x| *x = f(*x));
		mat
	}

	pub fn new(shape: (usize, usize), data: Vec<f64>) -> Self {
		assert_eq!(shape.0 * shape.1, data.len());
		Self { shape, data }
	}
	pub fn zero(shape: (usize, usize)) -> Self {
		Self::new(shape, vec![0.0; shape.0 * shape.1])
	}
	pub fn row(&self, row: usize) -> MatrixVectorView<'_> {
		MatrixVectorView {
			mat: self,
			offset: row * self.shape.1,
			step: 1,
			len: self.shape.1,
		}
	}
	pub fn column(&self, col: usize) -> MatrixVectorView<'_> {
		MatrixVectorView {
			mat: self,
			offset: col,
			step: self.shape.1,
			len: self.shape.0,
		}
	}
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
		self.data.iter_mut()
	}
	pub fn dot(&self, other: &Matrix) -> Matrix {
		assert_eq!(self.shape.1, other.shape.0);
		let (rows, cols) = (self.shape.0, other.shape.1);
		let mut data = vec![0.0; rows * cols];

		for i in 0..rows {
			for j in 0..cols {
				data[i * cols + j] = self.row(i).iter()
					.zip(other.column(j))
					.map(|(a, b)| a * b)
					.sum();
			}
		}
		Matrix::new((rows, cols), data)
	}
	/// dot(self, other^T)
	pub fn dot_transpose(&self, other: &Matrix) -> Matrix {
		assert_eq!(self.shape.1, other.shape.1);
		let (rows, cols) = (self.shape.0, other.shape.0);
		let mut data = vec![0.0; rows * cols];

		for i in 0..rows {
			for j in 0..cols {
				data[i * cols + j] = self.row(i).iter()
					.zip(other.row(j))
					.map(|(a, b)| a * b)
					.sum();
			}
		}
		Matrix::new((rows, cols), data)
	}
	/// dot(self^T, other)
	pub fn transpose_dot(&self, other: &Matrix) -> Matrix {
		assert_eq!(self.shape.0, other.shape.0);
		let (rows, cols) = (self.shape.1, other.shape.1);
		let mut data = vec![0.0; rows * cols];

		for i in 0..rows {
			for j in 0..cols {
				data[i * cols + j] = self.column(i).iter()
					.zip(other.column(j))
					.map(|(a, b)| a * b)
					.sum();
			}
		}
		Matrix::new((rows, cols), data)
	}
	pub fn transpose(&self) -> Matrix {
		let (cols, rows) = self.shape;
		let mut data = vec![0.0; cols * rows];

		for i in 0..rows {
			for j in 0..cols {
				data[i * cols + j] = self.get(j, i);
			}
		}
		Matrix::new((rows, cols), data)
	}
	pub fn get(&self, row: usize, col: usize) -> f64 {
		self.data[row * self.shape.1 + col]
	}
	pub fn copy_from_slice(&mut self, slice: &[f64]) {
		assert_eq!(self.data.len(), slice.len());
		self.data.copy_from_slice(slice);
	}
}
impl From<Vec<f64>> for Matrix {
	fn from(data: Vec<f64>) -> Self {
		let shape = (data.len(), 1);
		Self::new(shape, data)
	}
}
impl AddAssign<&Matrix> for Matrix {
	fn add_assign(&mut self, other: &Matrix) {
		assert_eq!(self.shape, other.shape);
		self.iter_mut().zip(other.data.iter())
			.for_each(|(a, b)| *a += b);
	}
}
impl AddAssign<Matrix> for Matrix {
	fn add_assign(&mut self, other: Matrix) {
		self.add_assign(&other);
	}
}
impl Add<&Matrix> for Matrix {
	type Output = Matrix;

	fn add(mut self, other: &Matrix) -> Self::Output {
		assert_eq!(self.shape, other.shape);
		self.add_assign(other);
		self
	}
}
impl Add<Matrix> for Matrix {
	type Output = Matrix;

	fn add(self, other: Matrix) -> Self::Output {
		self + &other
	}
}
impl SubAssign<&Matrix> for Matrix {
	fn sub_assign(&mut self, other: &Matrix) {
		assert_eq!(self.shape, other.shape);
		self.iter_mut().zip(other.data.iter())
			.for_each(|(a, b)| *a -= b);
	}
}
impl SubAssign<Matrix> for Matrix {
	fn sub_assign(&mut self, other: Matrix) {
		self.sub_assign(&other);
	}
}
impl Sub<&Matrix> for Matrix {
	type Output = Matrix;

	fn sub(mut self, other: &Matrix) -> Self::Output {
		self.sub_assign(other);
		self
	}
}
impl Sub<Matrix> for Matrix {
	type Output = Matrix;

	fn sub(self, other: Matrix) -> Self::Output {
		self - &other
	}
}
impl MulAssign<&Matrix> for Matrix {
	fn mul_assign(&mut self, other: &Matrix) {
		assert_eq!(self.shape, other.shape);
		self.iter_mut().zip(other.data.iter())
			.for_each(|(a, b)| *a *= b);
	}
}
impl Mul<&Matrix> for Matrix {
	type Output = Matrix;

	fn mul(mut self, other: &Matrix) -> Self::Output {
		self.mul_assign(other);
		self
	}
}
impl Mul<Matrix> for Matrix {
	type Output = Matrix;

	fn mul(self, other: Matrix) -> Self::Output {
		self * &other
	}
}
impl Mul<f64> for Matrix {
	type Output = Matrix;

	fn mul(mut self, scalar: f64) -> Self::Output {
		self.iter_mut().for_each(|x| *x *= scalar);
		self
	}
}
impl Display for Matrix {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "Matrix{{{}x{}}}", self.shape.0, self.shape.1)
	}
}

#[derive(Clone)]
pub struct MatrixVectorView<'a> {
	mat: &'a Matrix,
	offset: usize,
	step: usize,
	len: usize,
}
impl MatrixVectorView<'_> {
	pub fn len(&self) -> usize {
		self.len
	}
	pub fn iter(&self) -> MatrixVectorViewIterator {
		MatrixVectorViewIterator {
			view: self.clone(),
			index: 0,
		}
	}
	pub fn argmax(&self) -> usize {
		self.iter().enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
			.unwrap()
			.0
	}
}
impl Index<usize> for MatrixVectorView<'_> {
	type Output = f64;

	fn index(&self, index: usize) -> &Self::Output {
		if index >= self.len {
			panic!("index out of bounds");
		}
		&self.mat.data[self.offset + index * self.step]
	}
}
impl<'a> IntoIterator for MatrixVectorView<'a> {
	type Item = f64;
	type IntoIter = MatrixVectorViewIterator<'a>;

	fn into_iter(self) -> Self::IntoIter {
		MatrixVectorViewIterator {
			view: self,
			index: 0,
		}
	}
}

pub struct MatrixVectorViewIterator<'a> {
	view: MatrixVectorView<'a>,
	index: usize,
}
impl Iterator for MatrixVectorViewIterator<'_> {
	type Item = f64;

	fn next(&mut self) -> Option<Self::Item> {
		if self.index < self.view.len() {
			let value = self.view[self.index];
			self.index += 1;
			Some(value)
		} else {
			None
		}
	}
}
