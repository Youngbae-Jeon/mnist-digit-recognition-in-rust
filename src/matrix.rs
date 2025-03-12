use std::{fmt::Display, ops::{Add, AddAssign, Index, Mul, MulAssign, Sub, SubAssign}};

#[derive(Clone)]
pub struct Matrix {
	pub shape: (usize, usize),
	data: Vec<f32>,
}
impl Matrix {
	pub fn update(mut mat: Matrix, f: fn(f32) -> f32) -> Matrix {
		mat.iter_mut()
			.for_each(|x| *x = f(*x));
		mat
	}

	pub fn new(shape: (usize, usize), data: Vec<f32>) -> Self {
		assert_eq!(shape.0 * shape.1, data.len());
		Self { shape, data }
	}
	pub fn zero(shape: (usize, usize)) -> Self {
		Self::new(shape, vec![0.0; shape.0 * shape.1])
	}
	pub fn get(&self, row: usize, col: usize) -> f32 {
		self.data[row * self.shape.1 + col]
	}
	pub fn row(&self, row: usize) -> MatrixVectorView<'_> {
		MatrixVectorView::new_for_row(self, row)
	}
	pub fn column(&self, col: usize) -> MatrixVectorView<'_> {
		MatrixVectorView::new_for_column(self, col)
	}
	pub fn iter(&self) -> impl Iterator<Item = &f32> {
		self.data.iter()
	}
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
		self.data.iter_mut()
	}
	pub fn dot(&self, other: &Matrix) -> Matrix {
		assert_eq!(self.shape.1, other.shape.0);
		let (rows, cols) = (self.shape.0, other.shape.1);
		let mut data = vec![0.0; rows * cols];

		for i in 0..rows {
			for j in 0..cols {
				data[i * cols + j] = self.row(i).dot(&other.column(j));
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
				data[i * cols + j] = self.row(i).dot(&other.row(j));
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
				data[i * cols + j] = self.column(i).dot(&other.column(j));
			}
		}
		Matrix::new((rows, cols), data)
	}
	#[allow(dead_code)]
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
	pub fn norm(&self) -> f32 {
		self.iter().map(|x| x.powi(2)).sum::<f32>().sqrt()
	}
}
impl From<Vec<f32>> for Matrix {
	fn from(data: Vec<f32>) -> Self {
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
impl MulAssign<f32> for Matrix {
	fn mul_assign(&mut self, scalar: f32) {
		self.iter_mut().for_each(|x| *x *= scalar);
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
impl Mul<f32> for Matrix {
	type Output = Matrix;

	fn mul(mut self, scalar: f32) -> Self::Output {
		self.mul_assign(scalar);
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
	data: &'a [f32],
	offset: usize,
	step: usize,
	len: usize,
}
impl MatrixVectorView<'_> {
	pub fn new_for_row(mat: &Matrix, row_index: usize) -> MatrixVectorView<'_> {
		MatrixVectorView {
			data: &mat.data,
			offset: row_index * mat.shape.1,
			step: 1,
			len: mat.shape.1,
		}
	}
	pub fn new_for_column(mat: &Matrix, col_index: usize) -> MatrixVectorView<'_> {
		MatrixVectorView {
			data: &mat.data,
			offset: col_index,
			step: mat.shape.1,
			len: mat.shape.0,
		}
	}
	#[inline]
	pub fn len(&self) -> usize {
		self.len
	}
	pub fn iter(&self) -> MatrixVectorViewIterator {
		self.clone().into_iter()
	}
	pub fn dot(&self, other: &MatrixVectorView) -> f32 {
		assert_eq!(self.len, other.len);
		let mut sum = 0.0;
		for i in 0..self.len {
			sum += self[i] * other[i];
		}
		sum
	}
	pub fn argmax(&self) -> usize {
		self.iter().enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
			.map_or(0, |(i, _)| i)
	}
}
impl Index<usize> for MatrixVectorView<'_> {
	type Output = f32;

	#[inline]
	fn index(&self, index: usize) -> &Self::Output {
		&self.data[self.offset + index * self.step]
	}
}
impl<'a> IntoIterator for MatrixVectorView<'a> {
	type Item = f32;
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
	type Item = f32;

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
