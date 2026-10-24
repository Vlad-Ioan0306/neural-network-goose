// bias.goose — bias vector helpers built on matrix::Mat.
// A bias is a 1 x cols row vector added to every row of a layer's output.

namespace bias;

import matrix from "../matrix";

// b_init(size) returns a 1 x size zero bias vector.
fn b_init(size: u64) -> matrix::Mat {
	let m, _ = matrix::mat(1, size, []);
	return m;
}

// b_set(b, col, v) copies slice v into column `col` of the bias row.
// Returns false on bounds failure; matrix is untouched.
fn b_set(b: matrix::Mat&, col: u64, v: f64[:]) -> bool {
	if col >= b.cols { return false; }
	if v.len as u64 > b.cols - col { return false; }
	var i = 0;
	while i < v.len {
		let idx = col + i as u64;
		b.data[idx] = v[i];
		i++;
	}
	return true;
}

// b_add(m, b) adds bias b to every row of m. Returns (zero Mat, false) if b is not a 1 x m.cols row vector.
fn b_add(m: matrix::Mat&, b: matrix::Mat&) -> matrix::Mat, bool {
	if b.rows != 1 || b.cols != m.cols { return matrix::Mat { rows: 0, cols: 0, data: [] }, false; }
	var out: matrix::Mat = matrix::mat_cpy(m);
	for i in m.rows {
		for j in m.cols {
			let base = i * m.cols + j;
			out.data[base] = out.data[base] + b.data[j];
		}
	}
	return out, true;
}

// b_map<F>(b) applies f to every element of b in place.
// Call as: b_map(b) { x => x * 0.1 };
fn b_map<F>(b: matrix::Mat&) {
	matrix::mat_map(b)
}

// b_cpy(b) returns a deep copy of b.
fn b_cpy(b: matrix::Mat&) -> matrix::Mat {
	matrix::mat_cpy(b)
}
