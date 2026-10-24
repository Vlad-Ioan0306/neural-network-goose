// weights.goose — weight matrix helpers built on matrix::Mat.
// rows = out features, cols = in features.

namespace weights;

import matrix from "../matrix";

// w_init(rows, cols, seed) returns a randomly initialized rows x cols weight matrix.
// Same seed -> same matrix.
fn w_init(rows: u64, cols: u64, seed: u64) -> matrix::Mat {
	matrix::mat_gen(rows, cols, seed)
}

// w_set(w, row, col, v) copies slice v into row `row` at column `col`.
// Returns false on bounds failure; matrix is untouched.
fn w_set(w: matrix::Mat&, row: u64, col: u64, v: f64[:]) -> bool {
	if row >= w.rows { return false; }
	if col >= w.cols { return false; }
	let base = row * w.cols + col;
	if base + v.len as u64 > w.rows * w.cols { return false; }
	var i = 0;
	while i < v.len {
		let idx = base + i as u64;
		w.data[idx] = v[i];
		i++;
	}
	return true;
}

// w_map<F>(w) applies f to every element of w in place.
// Call as: w_map(w) { x => x * 0.01 };
fn w_map<F>(w: matrix::Mat&) {
	matrix::mat_map(w)
}

// w_cpy(w) returns a deep copy of w.
fn w_cpy(w: matrix::Mat&) -> matrix::Mat {
	matrix::mat_cpy(w)
}

// w_apply(w, input) returns w * input. Returns (zero Mat, false) if w.cols != input.rows.
fn w_apply(w: matrix::Mat&, input: matrix::Mat&) -> matrix::Mat, bool {
	if w.cols != input.rows { return matrix::Mat { rows: 0, cols: 0, data: [] }, false; }
	let c, ok = matrix::matmul(w, input);
	return c, ok;
}
