// matrix.goose — row-major f64 matrices as flat grow-only arrays.
// All matrices are stored in row-major order: element (r, c) lives at
// data[r * cols + c]. matmul transposes b once so every dot product walks
// two contiguous rows.
// Any function that can fail returns (result, bool); false means the
// operation did not happen and the result is a zero Mat / empty slice.

namespace matrix;

import std;
import math;
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
fn transpose(m: const Mat&) -> Mat {
    var data: f64[>..] = [];
    for i in m.cols {
        for j in m.rows {
            data.push(m.data[j * m.cols + i]);
        }
    }
    return Mat { rows: m.cols, cols: m.rows, data: data };
}

// mat_cpy(m) returns a deep copy of m.
fn mat_cpy(m: const Mat&) -> Mat {
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
fn mat_print(m: const Mat&) {
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
fn mat_sum_rows(m: const Mat&) -> f64[>..] {
    var out: f64[>..] = [];
    for i in m.rows {
        var s = 0.0;
        for j in m.cols { s += m.data[i * m.cols + j]; }
        out.push(s);
    }
    return out;
}

// mat_sum_cols(m) returns the sum of each column, length m.cols.
fn mat_sum_cols(m: const Mat&) -> f64[>..] {
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
fn mat_add(a: const Mat&, b: const Mat&) -> Mat, bool {
    if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
    var data: f64[>..] = [];
    for t in a.rows * a.cols { data.push(a.data[t] + b.data[t]); }
    return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// mat_mul(a, b) returns a * b elementwise (Hadamard). Returns (zero Mat, false) on shape mismatch.
fn mat_mul(a: const Mat&, b: const Mat&) -> Mat, bool {
    if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
    var data: f64[>..] = [];
    for t in a.rows * a.cols { data.push(a.data[t] * b.data[t]); }
    return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// mat_div(a, b) returns a / b elementwise. Returns (zero Mat, false) on shape
// mismatch or if any element of b is zero.
fn mat_div(a: const Mat&, b: const Mat&) -> Mat, bool {
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
fn mat_row(m: const Mat&, row: u64) -> f64[>..], bool {
    if row >= m.rows { return [], false; }
    let base = row * m.cols;
    var out: f64[>..] = [];
    for j in m.cols { out.push(m.data[base + j]); }
    return out, true;
}

// mat_col(m, col) returns a copy of column `col`. Returns ([], false) on bad index.
fn mat_col(m: const Mat&, col: u64) -> f64[>..], bool {
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
fn matmul(a: const Mat&, b: const Mat&) -> Mat, bool {
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

// --- save / load -----------------------------------------------------------

fn write_u64(out: u8[>..]&, v: u64) {
    if v == 0 { out.push('0'); return; }
    var tmp: u8[>..] = [];
    var n = v;
    while n > 0 {
        tmp.push(('0' + (n % 10) as u8));
        n = n / 10;
    }
    var i = tmp.len;
    while i > 0 {
        i = i - 1;
        out.push(tmp[i]);
    }
}

fn parse_f64(text: u8[:], pos: i64&) -> f64 {
    var i = pos;
    var neg = false;
    if i < text.len && text[i] == '-' { neg = true; i++; }
    var v = 0.0;
    while i < text.len && text[i] >= '0' && text[i] <= '9' {
        v = v * 10.0 + (text[i] - '0');
        i++;
    }
    if i < text.len && text[i] == '.' {
        i++;
        var mult = 0.1;
        while i < text.len && text[i] >= '0' && text[i] <= '9' {
            v = v + (text[i] - '0') * mult;
            mult = mult * 0.1;
            i++;
        }
    }
    if i < text.len && (text[i] == 'e' || text[i] == 'E') {
        i++;
        var neg_exp = false;
        if i < text.len && text[i] == '-' { neg_exp = true; i++; }
        if i < text.len && text[i] == '+' { i++; }
        var exp: u64 = 0;
        while i < text.len && text[i] >= '0' && text[i] <= '9' {
            exp = exp * 10 + (text[i] - '0');
            i++;
        }
        var factor = 1.0;
        var e = exp;
        while e > 0 { factor = factor * 10.0; e = e - 1; }
        if neg_exp { v = v / factor; } else { v = v * factor; }
    }
    pos = i;
    if neg { v = -v; }
    return v;
}

fn mat_save(m: const Mat&, path: const u8[:]) -> bool {
    var out: u8[>..] = [];
    write_u64(out, m.rows);
    out.push(' ');
    write_u64(out, m.cols);
    out.push('\n');
    for t in m.rows * m.cols {
        format_flt(out, m.data[t], 17);
        out.push('\n');
    }
    return write_file_atomic(path, out);
}

fn mat_load(path: const u8[:]) -> Mat, bool {
    var text: u8[>..] = [];
    if !read_file(path, text) { return Mat { rows: 0, cols: 0, data: [] }, false; }
    var i: i64 = 0;
    var rows: u64 = 0;
    while i < text.len && text[i] >= '0' && text[i] <= '9' {
        rows = rows * 10 + (text[i] - '0');
        i++;
    }
    if i >= text.len || text[i] != ' ' { return Mat { rows: 0, cols: 0, data: [] }, false; }
    i++;
    var cols: u64 = 0;
    while i < text.len && text[i] >= '0' && text[i] <= '9' {
        cols = cols * 10 + (text[i] - '0');
        i++;
    }
    if i >= text.len || text[i] != '\n' { return Mat { rows: 0, cols: 0, data: [] }, false; }
    i++;
    var data: f64[>..] = [];
    for t in rows * cols {
        while i < text.len && text[i] <= ' ' { i++; }
        var v = parse_f64(text, i);
        data.push(v);
    }
    return Mat { rows: rows, cols: cols, data: data }, true;
}

// --- dense backward ---------------------------------------------------------

// dense_backward(w, x, dz) computes gradients given output gradient dz.
// dW = dz · x^T, db = column sums of dz, dx = w^T · dz.
// Returns (dW, db, dx, bool). false on dimension mismatch.
fn dense_backward(w: const Mat&, x: const Mat&, dz: const Mat&) -> Mat, f64[>..], Mat, bool {
	var xt = transpose(x);
	var dw, dok = matmul(dz, xt);
	if !dok { return Mat { rows: 0, cols: 0, data: [] }, [], Mat { rows: 0, cols: 0, data: [] }, false; }
	var db = mat_sum_cols(dz);
	var wt = transpose(w);
	var dx, dok2 = matmul(wt, dz);
	if !dok2 { return Mat { rows: 0, cols: 0, data: [] }, [], Mat { rows: 0, cols: 0, data: [] }, false; }
	return dw, db, dx, true;
}

// --- loss functions ---------------------------------------------------------

// mat_bce(pred, target) returns binary cross-entropy loss.
// Returns (mean BCE, false) on shape mismatch.
// Clips predictions to [eps, 1-eps] to avoid log(0).
fn mat_bce(pred: const Mat&, target: const Mat&) -> f64, bool {
	if pred.rows != target.rows || pred.cols != target.cols { return 0.0, false; }
	var eps = 1e-15;
	var s = 0.0;
	for t in pred.rows * pred.cols {
		var p = pred.data[t];
		if p < eps { p = eps; }
		if p > 1.0 - eps { p = 1.0 - eps; }
		s = s - (target.data[t] * log(p) + (1.0 - target.data[t]) * log(1.0 - p));
	}
	return s / (pred.rows * pred.cols), true;
}

// mat_acc(pred, target) returns binary accuracy (pred > 0.5 == target).
// Returns (accuracy, false) on shape mismatch.
fn mat_acc(pred: const Mat&, target: const Mat&) -> f64, bool {
	if pred.rows != target.rows || pred.cols != target.cols { return 0.0, false; }
	var correct: f64 = 0.0;
	var total: u64 = pred.rows * pred.cols;
	for t in total {
		var p = pred.data[t];
		if p < 0.0 { p = 0.0; }
		if p > 1.0 { p = 1.0; }
		var pred_bit = if p > 0.5 { 1.0 } else { 0.0 };
		if pred_bit == target.data[t] { correct = correct + 1.0; }
	}
	return correct / total as f64, true;
}

// --- weight initialization --------------------------------------------------

// mat_xavier_init(rows, cols) returns a matrix initialized with Xavier/Glorot
// uniform: U[-limit, limit] where limit = sqrt(6 / (rows + cols)).
fn mat_xavier_init(rows: u64, cols: u64) -> Mat {
	var limit = sqrt(6.0 / (rows + cols) as f64);
	var r = rng { random_seed() };
	var data: f64[>..] = [];
	for t in rows * cols {
		var v = rand_flt(r) * 2.0 * limit - limit;
		data.push(v);
	}
	return Mat { rows: rows, cols: cols, data: data };
}

// mat_he_init(rows, cols) returns a matrix initialized with He uniform:
// U[-limit, limit] where limit = sqrt(6 / rows).
fn mat_he_init(rows: u64, cols: u64) -> Mat {
	var limit = sqrt(6.0 / rows as f64);
	var r = rng { random_seed() };
	var data: f64[>..] = [];
	for t in rows * cols {
		var v = rand_flt(r) * 2.0 * limit - limit;
		data.push(v);
	}
	return Mat { rows: rows, cols: cols, data: data };
}

// --- elementwise binary (same shape) ----------------------------------------

// mat_sub(a, b) returns a - b elementwise. Returns (zero Mat, false) on shape mismatch.
fn mat_sub(a: const Mat&, b: const Mat&) -> Mat, bool {
	if a.rows != b.rows || a.cols != b.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
	var data: f64[>..] = [];
	for t in a.rows * a.cols { data.push(a.data[t] - b.data[t]); }
	return Mat { rows: a.rows, cols: a.cols, data: data }, true;
}

// mat_axpy(w, scale, delta) does w += scale * delta in place. Returns false on shape mismatch.
fn mat_axpy(w: Mat&, scale: f64, delta: const Mat&) -> bool {
	if w.rows != delta.rows || w.cols != delta.cols { return false; }
	for t in w.rows * w.cols { w.data[t] = w.data[t] + scale * delta.data[t]; }
	return true;
}

// --- dense train step -------------------------------------------------------

// mat_mse_loss(pred, target) returns mean squared error. Returns false on shape mismatch.
fn mat_mse_loss(pred: const Mat&, target: const Mat&) -> f64, bool {
	if pred.rows != target.rows || pred.cols != target.cols { return 0.0, false; }
	var s = 0.0;
	for t in pred.rows * pred.cols {
		var d = pred.data[t] - target.data[t];
		s = s + d * d;
	}
	return s / (pred.rows * pred.cols), true;
}

// dense_train_step(w, b, x, target, lr) runs forward, backward, and updates
// w and b in place by lr * gradient. Returns (loss, false) on failure.
fn dense_train_step(w: Mat&, b: Mat&, x: Mat&, target: Mat&, lr: f64) -> f64, bool {
	var z, ok = dense_forward(w, x, b);
	if !ok { return 0.0, false; }

	var diff, dok = mat_sub(z, target);
	if !dok { return 0.0, false; }

	var n = (z.rows * z.cols) as f64;
	mat_map(diff) { v => 2.0 * v / n };
	var dz = diff;

	let dw, db, dx, dok2 = dense_backward(w, x, dz);
	if !dok2 { return 0.0, false; }

	if !mat_axpy(w, -lr, dw) { return 0.0, false; }
	for t in b.rows * b.cols { b.data[t] = b.data[t] + (-lr) * db[t]; }

	return mat_mse_loss(z, target), true;
}

// --- dense layer -----------------------------------------------------------

// layer_forward(weights, biases, x) chains dense_forward + relu through every layer.
// weights[i] and biases[i] are references to layer i's parameters.
// Returns the final output, or zero Mat + false on dimension mismatch.
fn layer_forward(weights: const Mat&[:], biases: const Mat&[:], x: Mat&) -> Mat, bool {
    var z, ok = dense_forward(weights[0], x, biases[0]);
    if !ok { return Mat { rows: 0, cols: 0, data: [] }, false; }
    matrix::mat_map(z) { v => if v < 0.0 { 0.0 } else { v } };
    
    var i = 1;
    while i < weights.len {
        var z2, ok2 = dense_forward(weights[i], z, biases[i]);
        if !ok2 { return Mat { rows: 0, cols: 0, data: [] }, false; }
        matrix::mat_map(z2) { v => if v < 0.0 { 0.0 } else { v } };
        z = copy(z2);
        i++;
    }
    return z, true;
}

// --- dense forward ----------------------------------------------------------
fn dense_forward(w: const Mat&, x: const Mat&, b: const Mat&) -> Mat, bool {
    var z, ok = matmul(w, x);
    if !ok { return Mat { rows: 0, cols: 0, data: [] }, false; }
    if b.rows != 1 || b.cols != z.cols { return Mat { rows: 0, cols: 0, data: [] }, false; }
    for i in z.rows {
        for j in z.cols {
            let base = i * z.cols + j;
            z.data[base] = z.data[base] + b.data[j];
        }
    }
    return z, true;
}
