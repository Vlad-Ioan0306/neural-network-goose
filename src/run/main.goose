// run/main.goose — demonstration of the Neural Network library
// Run with: goose build run/main.goose && ./run/main

import matrix from "../code/include/matrix";
import weights from "../code/include/weights";
import bias from "../code/include/bias";
import functions from "../code/include/functions";

fn approx(a: f64, b: f64, eps: f64) -> bool {
	var d = a - b;
	if d < 0.0 { d = -d; }
	return d < eps;
}

fn main() {
	print("=== Neural Network Library Demo ===");
	print("");

	// --- 1. Build a weight matrix (2x3) and bias vector (1x2) ---
	print("Step 1: Create weight matrix and bias vector");
	var wback: f64[>..] = [];
	wback.push(1.0); wback.push(2.0); wback.push(3.0);
	wback.push(4.0); wback.push(5.0); wback.push(6.0);
	var w_row0: f64[:] = wback[0..3];
	var w_row1: f64[:] = wback[3..6];
	var w_data: f64[:][>..] = []; w_data.push(w_row0); w_data.push(w_row1);
	let w, wok = matrix::mat(2, 3, copy(w_data));
	if !wok { print("FAIL: could not create weight matrix"); return; }

	var bback: f64[>..] = [];
	bback.push(0.1); bback.push(0.2);
	var b_row0: f64[:] = bback[0..2];
	var b_data: f64[:][>..] = []; b_data.push(b_row0);
	let b, bok = matrix::mat(1, 2, copy(b_data));
	if !bok { print("FAIL: could not create bias matrix"); return; }

	print("  W = [[1,2,3],[4,5,6]]  (2x3)");
	print("  b = [[0.1,0.2]]        (1x2)");
	print("");

	// --- 2. Create input (3x2) ---
	print("Step 2: Create input matrix (3x2)");
	var xback: f64[>..] = [];
	xback.push(1.0); xback.push(2.0);
	xback.push(3.0); xback.push(4.0);
	xback.push(5.0); xback.push(6.0);
	var x_row0: f64[:] = xback[0..2];
	var x_row1: f64[:] = xback[2..4];
	var x_row2: f64[:] = xback[4..6];
	var x_data: f64[:][>..] = []; x_data.push(x_row0); x_data.push(x_row1); x_data.push(x_row2);
	let x, xok = matrix::mat(3, 2, copy(x_data));
	if !xok { print("FAIL: could not create input matrix"); return; }

	print("  X = [[1,2],[3,4],[5,6]]  (3x2)");
	print("");

	// --- 3. Forward pass: z = W @ x + b ---
	print("Step 3: Forward pass z = W @ x + b");
	let z, zok = matrix::dense_forward(w, x, b);
	if !zok { print("FAIL: matmul failed"); return; }
	matrix::mat_print(z);
	print("");

	// --- 4. Apply ReLU activation ---
	print("Step 4: Apply ReLU activation");
	functions::mat_relu(z);
	matrix::mat_print(z);
	print("");

	// --- 5. Backpropagation ---
	print("Step 5: Backpropagation");
	var dzback: f64[>..] = [];
	dzback.push(1.0); dzback.push(2.0);
	dzback.push(3.0); dzback.push(4.0);
	var dz_row0: f64[:] = dzback[0..2];
	var dz_row1: f64[:] = dzback[2..4];
	var dz_data: f64[:][>..] = []; dz_data.push(dz_row0); dz_data.push(dz_row1);
	let dz, dzok = matrix::mat(2, 2, copy(dz_data));
	if !dzok { print("FAIL: could not create gradient matrix"); return; }

	let dw, db, dx, dok = matrix::dense_backward(w, x, dz);
	if !dok { print("FAIL: backward pass failed"); return; }

	print("  dW shape: rows=");
	print(dw.rows);
	print(" cols=");
	print(dw.cols);
	print("  db shape: [");
	print(dw.rows);
	print("]");
	print("  dx shape: rows=");
	print(dx.rows);
	print(" cols=");
	print(dx.cols);
	print("");

	// --- 6. Training step ---
	print("Step 6: Training step (SGD with lr=0.01)");
	var tyback: f64[>..] = [];
	tyback.push(10.0); tyback.push(20.0);
	tyback.push(30.0); tyback.push(40.0);
	var ty_row0: f64[:] = tyback[0..2];
	var ty_row1: f64[:] = tyback[2..4];
	var ty_data: f64[:][>..] = []; ty_data.push(ty_row0); ty_data.push(ty_row1);
	let ty, tyok = matrix::mat(2, 2, copy(ty_data));
	if !tyok { print("FAIL: could not create target matrix"); return; }

	let loss1, lok1 = matrix::dense_train_step(w, b, x, ty, 0.01);
	if !lok1 { print("FAIL: dense_train_step failed"); return; }
	print("  Learning rate: 0.01");
	print("  Loss after step: ");
	print(loss1);
	print("  Weight updated via w += -lr * dW");
	print("  Bias updated via b += -lr * db");
	print("");

	// --- 7. Save and load ---
	print("Step 7: Save and load matrix");
	let save_ok = matrix::mat_save(w, "/tmp/test_matrix.txt");
	if !save_ok { print("FAIL: mat_save failed"); return; }
	let load_mat, lok = matrix::mat_load("/tmp/test_matrix.txt");
	if !lok { print("FAIL: mat_load failed"); return; }
	print("  Saved W to /tmp/test_matrix.txt and loaded back:");
	matrix::mat_print(load_mat);
	print("");

	print("=== Demo complete ===");
}
