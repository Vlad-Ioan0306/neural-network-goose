// matrix.goose — row-major f64 matrices as flat grow-only arrays.
// All matrices are stored in row-major order: element (r, c) lives at
// data[r * cols + c]. matmul transposes b once so every dot product walks
// two contiguous rows.
// Any function that can fail returns (result, bool); false means the
// operation did not happen and the result is a zero Mat / empty slice.

namespace matrix;

import std;
import os;

struct Mat {
	rows: u64,
	cols: u64,
	data: f64[>..],
}

// --- construction -----------------------------------------------------------

// mat(rows, cols, data) returns a rows x cols matrix built from a 2D
// f64[:][>..] array, flattened row-major into Mat.data. Empty data produces
// a zero-filled matrix. Returns false if data doesn't match rows x cols.
fn mat(rows: u64, cols: u64, data: f64[:][>..]) -> Mat, bool {
	if data.len as u64 == 0 {
		var d: f64[>..] = [];
		for t in rows * cols { d.push(0.0); }
		return Mat { rows: rows, cols: cols, data: d }, true;
	}
	if data.len as u64 != rows { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var d: f64[>..] = [];
	var i = 0;
	for row in data {
		if row.len as u64 != cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
		var j = 0;
		for val in row {
			d.push(val);
			j++;
		}
		i++;
	}
	return Mat { rows: rows, cols: cols, data: d }, true;
}

// --- transformation ---------------------------------------------------------

// transpose(m) returns the transpose of m.
fn transpose(m: Mat&) -> Mat {
	var data: f64[>..] = [];
	for i in m.cols {
		for j in m.rows {
			data.push(m.data[j * m.cols + i]);
		}
	}
	return Mat { rows: m.cols, cols: m.rows, data: data };
}

// mat_cpy(m) returns a deep copy of m.
fn mat_cpy(m: Mat&) -> Mat {
	var data: f64[>..] = [];
	for t in m.rows * m.cols { data.push(m.data[t]); }
	return Mat { rows: m.rows, cols: m.cols, data: data };
}

// mat_gen(rows, cols, seed) returns a rows x cols matrix filled with random
// f64 values in [0, 1), seeded from the given seed. Same seed -> same matrix.
fn mat_gen(rows: u64, cols: u64, seed: u64) -> Mat {
	var r = rng { seed };
	var data: f64[>..] = [];
	for t in rows * cols { data.push(rand_flt(r)); }
	return Mat { rows: rows, cols: cols, data: data };
}

// --- printing ---------------------------------------------------------------

// mat_print(m) prints m row by row, space-separated.
fn mat_print(m: Mat&) {
	for i in m.rows {
		for j in m.cols {
			print(m.data[i * m.cols + j]);
			if j < m.cols - 1 { print(" "); }
		}
		print("\n");
	}
}

// --- elementwise map --------------------------------------------------------

// mat_map<F>(m, f) applies f to every element of m in place.
// Call as: mat_map(m) { x => x * 2.0 };
fn mat_map<F>(m: Mat&) {
	for t in m.rows * m.cols { m.data[t] = F(m.data[t]); }
}

// --- reductions -------------------------------------------------------------

// mat_sum_rows(m) returns the sum of each row, length m.rows.
fn mat_sum_rows(m: Mat&) -> f64[>..] {
	var out: f64[>..] = [];
	for i in m.rows {
		var s = 0.0;
		for j in m.cols { s += m.data[i * m.cols + j]; }
		out.push(s);
	}
	return out;
}

// mat_sum_cols(m) returns the sum of each column, length m.cols.
fn mat_sum_cols(m: Mat&) -> f64[>..] {
	var out: f64[>..] = [];
	for j in m.cols {
		var s = 0.0;
		for i in m.rows { s += m.data[i * m.cols + j]; }
		out.push(s);
	}
	return out;
}

// --- elementwise binary (same shape) ----------------------------------------

// mat_add(a, b) returns a + b elementwise. Returns (zero Mat, false) on shape mismatch.
fn mat_add(a: Mat&, b: Mat&) -> Mat, bool {
	if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for t in a.rows * a.cols { data.push(a.data[t] + b.data[t]); }
	return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// mat_mul(a, b) returns a * b elementwise (Hadamard). Returns (zero Mat, false) on shape mismatch.
fn mat_mul(a: Mat&, b: Mat&) -> Mat, bool {
	if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for t in a.rows * a.cols { data.push(a.data[t] * b.data[t]); }
	return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// mat_div(a, b) returns a / b elementwise. Returns (zero Mat, false) on shape
// mismatch or if any element of b is zero.
fn mat_div(a: Mat&, b: Mat&) -> Mat, bool {
	if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	for t in a.rows * a.cols {
		if b.data[t] == 0.0 { return Mat { rows: 0, cols: 0, data: [] }, false; }
	}
	var data: f64[>..] = [];
	for t in a.rows * a.cols { data.push(a.data[t] / b.data[t]); }
	return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// --- row / column access ----------------------------------------------------

// mat_row(m, row) returns a copy of row `row`. Returns ([], false) on bad index.
fn mat_row(m: Mat&, row: u64) -> f64[>..], bool {
	if row >= m.rows { return [], false; }
	let base = row * m.cols;
	var out: f64[>..] = [];
	for j in m.cols { out.push(m.data[base + j]); }
	return out, true;
}

// mat_col(m, col) returns a copy of column `col`. Returns ([], false) on bad index.
fn mat_col(m: Mat&, col: u64) -> f64[>..], bool {
	if col >= m.cols { return [], false; }
	var out: f64[>..] = [];
	for i in m.rows { out.push(m.data[i * m.cols + col]); }
	return out, true;
}

// --- reshape / concat -------------------------------------------------------

// mat_reshape(m, rows, cols) returns m reshaped. Returns (zero Mat, false) if rows*cols != len.
fn mat_reshape(m: Mat&, rows: u64, cols: u64) -> Mat, bool {
	if rows * cols != m.rows * m.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for t in m.rows * m.cols { data.push(m.data[t]); }
	return Mat { rows: rows, cols: cols, data: data }, true;
}

// mat_concat_rows(a, b) stacks b below a. Returns (zero Mat, false) if column counts differ.
fn mat_concat_rows(a: Mat&, b: Mat&) -> Mat, bool {
	if a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for t in a.rows * a.cols { data.push(a.data[t]); }
	for t in b.rows * b.cols { data.push(b.data[t]); }
	return Mat { rows: a.rows + b.rows, cols: a.cols, data: data }, true;
}

// mat_concat_cols(a, b) stacks b to the right of a. Returns (zero Mat, false) if row counts differ.
fn mat_concat_cols(a: Mat&, b: Mat&) -> Mat, bool {
	if a.rows != b.rows { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for i in a.rows {
		for j in a.cols { data.push(a.data[i * a.cols + j]); }
		for j in b.cols { data.push(b.data[i * b.cols + j]); }
	}
	return Mat { rows: a.rows, cols: a.cols + b.cols, data: data }, true;
}

// --- matrix multiply --------------------------------------------------------

// matmul(a, b) returns the matrix product a * b. Returns (zero Mat, false) if a.cols != b.rows.
fn matmul(a: Mat&, b: Mat&) -> Mat, bool {
	if a.cols != b.rows { return Mat { rows: 0, cols: 0, data: [] }, false; }
	let bt = transpose(b);
	var data: f64[>..] = [];
	for t in a.rows * b.cols { data.push(0.0); }
	for i in a.rows {
		let a_off = i * a.cols;
		let c_off = i * b.cols;
		for j in b.cols {
			let b_off = j * bt.cols;
			var s = 0.0;
			for k in a.cols {
				s += a.data[a_off + k] * bt.data[b_off + k];
			}
			data[c_off + j] = s;
		}
	}
	return Mat { rows: a.rows, cols: b.cols, data: data }, true;
}
