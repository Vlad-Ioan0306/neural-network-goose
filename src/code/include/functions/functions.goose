// functions.goose — elementwise non-linearities built on matrix::mat_map.
// Softmax also needs row sums (matrix::mat_sum_rows).

namespace functions;

import matrix from "../matrix";
import math;

// mat_relu(m) applies ReLU in place: x -> max(x, 0).
fn mat_relu(m: matrix::Mat&) {
	matrix::mat_map(m) { x => if x < 0.0 { 0.0 } else { x } };
}

// mat_sigmoid(m) applies the logistic sigmoid in place: x -> 1 / (1 + e^-x).
fn mat_sigmoid(m: matrix::Mat&) {
	matrix::mat_map(m) { x => 1.0 / (1.0 + exp(-x)) };
}

// mat_tanh(m) applies tanh in place: x -> tanh(x).
fn mat_tanh(m: matrix::Mat&) {
	matrix::mat_map(m) { x => tanh(x) };
}

// mat_softmax_rows(m) applies row-wise softmax in place: exp each element,
// then divide by its row sum. Returns false if any row sum is zero.
// Safe because exp(x) > 0 for all finite x, so every non-empty row sum is
// strictly positive unless exp underflows to 0.0 for very negative inputs.
fn mat_softmax_rows(m: matrix::Mat&) -> bool {
	matrix::mat_map(m) { x => exp(x) };
	let sums = matrix::mat_sum_rows(m);
	for i in m.rows {
		if sums[i] == 0.0 { return false; }
		for j in m.cols {
			let base = i * m.cols + j;
			m.data[base] = m.data[base] / sums[i];
		}
	}
	return true;
}
