// src/run/main.goose — Neural Network demo: XOR with 2->3->1 MLP
// Trains via backprop (SGD) and prints expected vs output at the end.
// Run with: goose build src/run/main.goose && ./src/run/main
//
// Layout convention (library requires w @ x):
//   X: 2x4  (2 features, 4 samples — each column is a sample)
//   Y: 1x4  (1 output, 4 samples)
//   W1: 3x2 (3 hidden, 2 inputs), b1: 3x1 (one bias per hidden neuron)
//   W2: 1x3 (1 output, 3 hidden), b2: 1x1 (one bias per output neuron)

import matrix from "../code/include/matrix";
import weights from "../code/include/weights";
import bias from "../code/include/bias";
import functions from "../code/include/functions";

fn main() {
	var failures: i64 = 0;
	var lr: f64 = 0.5;
	var epochs: u64 = 50000;

	print("=== XOR Neural Network (2 -> 3 -> 1) ===");
	print("");

	// XOR dataset (columns = samples)
	var xflat: f64[>..] = [];
	xflat.push(0.0); xflat.push(0.0); xflat.push(1.0); xflat.push(1.0);
	xflat.push(0.0); xflat.push(1.0); xflat.push(0.0); xflat.push(1.0);
	var x_row0: f64[:] = xflat[0..4];
	var x_row1: f64[:] = xflat[4..8];
	var xdata: f64[:][>..] = []; xdata.push(x_row0); xdata.push(x_row1);
	let X, xok = matrix::mat(2, 4, copy(xdata));
	if !xok { print("FAIL: could not create X"); failures++; return; }

	var yflat: f64[>..] = [];
	yflat.push(0.0); yflat.push(1.0); yflat.push(1.0); yflat.push(0.0);
	var y_row0: f64[:] = yflat[0..4];
	var ydata: f64[:][>..] = []; ydata.push(y_row0);
	let Y, yok = matrix::mat(1, 4, copy(ydata));
	if !yok { print("FAIL: could not create Y"); failures++; return; }

	// Initialize weights
	let W1 = matrix::mat_xavier_init(3, 2);

	var b1flat: f64[>..] = [];
	b1flat.push(0.0); b1flat.push(0.0); b1flat.push(0.0);
	var b1_row0: f64[:] = b1flat[0..1];
	var b1_row1: f64[:] = b1flat[1..2];
	var b1_row2: f64[:] = b1flat[2..3];
	var b1data: f64[:][>..] = []; b1data.push(b1_row0); b1data.push(b1_row1); b1data.push(b1_row2);
	let b1m, b1mok = matrix::mat(3, 1, copy(b1data));
	if !b1mok { print("FAIL: could not create b1"); failures++; return; }

	let W2 = matrix::mat_xavier_init(1, 3);

	var b2flat: f64[>..] = [];
	b2flat.push(0.0);
	var b2_col0: f64[:] = b2flat[0..1];
	var b2data: f64[:][>..] = []; b2data.push(b2_col0);
	let b2m, b2mok = matrix::mat(1, 1, copy(b2data));
	if !b2mok { print("FAIL: could not create b2"); failures++; return; }

	var w1 = W1;
	var w1_b = b1m;
	var w2 = W2;
	var w2_b = b2m;

	// Ones matrix (1x4) for sigmoid derivative: 1 - a2
	var onesflat: f64[>..] = [];
	onesflat.push(1.0); onesflat.push(1.0); onesflat.push(1.0); onesflat.push(1.0);
	var o_row0: f64[:] = onesflat[0..4];
	var onesdata: f64[:][>..] = []; onesdata.push(o_row0);
	let ones14, ones14ok = matrix::mat(1, 4, copy(onesdata));
	if !ones14ok { print("FAIL: ones14"); failures++; return; }

	// Initial predictions (before training)
	let z1_init, z1iok = matrix::dense_forward(w1, X, w1_b);
	if !z1iok { print("FAIL: forward L1 (init)"); failures++; return; }
	matrix::mat_map(z1_init) { x => if x < 0.0 { 0.0 } else { x } };
	let z2_init, z2iok = matrix::dense_forward(w2, z1_init, w2_b);
	if !z2iok { print("FAIL: forward L2 (init)"); failures++; return; }
	functions::mat_sigmoid(z2_init);

	print("--- Predictions BEFORE training ---");
	print("Input    Expected  Output");
	print("-------  --------  ------");
	print("0,0      0         ");
	print(z2_init.data[0]);
	print("0,1      1         ");
	print(z2_init.data[1]);
	print("1,0      1         ");
	print(z2_init.data[2]);
	print("1,1      0         ");
	print(z2_init.data[3]);
	print("");

	// Training loop
	print("Training...");
	print("epochs: ");
	print(epochs);
	print("");

	var e: u64 = 0;
	while e < epochs {
		let z1, z1ok = matrix::dense_forward(w1, X, w1_b);
		if !z1ok { print("FAIL: forward L1"); failures++; return; }
		matrix::mat_map(z1) { x => if x < 0.0 { 0.0 } else { x } };

		let z2, z2ok = matrix::dense_forward(w2, z1, w2_b);
		if !z2ok { print("FAIL: forward L2"); failures++; return; }
		functions::mat_sigmoid(z2);

		let diff2, dif2ok = matrix::mat_sub(z2, Y);
		if !dif2ok { print("FAIL: mat_sub diff2"); failures++; return; }
		let one_minus_a2, oma2ok = matrix::mat_sub(ones14, z2);
		if !oma2ok { print("FAIL: mat_sub one_minus_a2"); failures++; return; }
		let sig_prime, spok = matrix::mat_mul(one_minus_a2, z2);
		if !spok { print("FAIL: mat_mul sig_prime"); failures++; return; }
		let dz2, dz2ok = matrix::mat_mul(diff2, sig_prime);
		if !dz2ok { print("FAIL: mat_mul dz2"); failures++; return; }

		let dw2, db2, da1, dok2 = matrix::dense_backward(w2, z1, dz2);
		if !dok2 { print("FAIL: backward L2"); failures++; return; }

		matrix::mat_map(z1) { x => if x > 0.0 { 1.0 } else { 0.0 } };
		let dz1, dz1ok = matrix::mat_mul(da1, z1);
		if !dz1ok { print("FAIL: mat_mul dz1"); failures++; return; }

		let dw1, db1, dx, dok1 = matrix::dense_backward(w1, X, dz1);
		if !dok1 { print("FAIL: backward L1"); failures++; return; }

		matrix::mat_map(dw1) { x => -x * lr };
		let new_w1, nw1ok = matrix::mat_add(w1, dw1);
		if !nw1ok { print("FAIL: mat_add W1"); failures++; return; }
		w1 = copy(new_w1);

		matrix::mat_map(dw2) { x => -x * lr };
		let new_w2, nw2ok = matrix::mat_add(w2, dw2);
		if !nw2ok { print("FAIL: mat_add W2"); failures++; return; }
		w2 = copy(new_w2);

		// b1 is 3x1, db1 is f64[:] len 3
		var new_b1data: f64[>..] = [];
		new_b1data.push(w1_b.data[0] - lr * db1[0]);
		new_b1data.push(w1_b.data[1] - lr * db1[1]);
		new_b1data.push(w1_b.data[2] - lr * db1[2]);
		var nb1_row0: f64[:] = new_b1data[0..1];
		var nb1_row1: f64[:] = new_b1data[1..2];
		var nb1_row2: f64[:] = new_b1data[2..3];
		var nb1data: f64[:][>..] = []; nb1data.push(nb1_row0); nb1data.push(nb1_row1); nb1data.push(nb1_row2);
		let new_b1, nb1ok = matrix::mat(3, 1, copy(nb1data));
		if !nb1ok { print("FAIL: new b1"); failures++; return; }
		w1_b = copy(new_b1);

		// b2 is 1x1, db2 is f64[:] len 1
		var new_b2data: f64[>..] = [];
		new_b2data.push(w2_b.data[0] - lr * db2[0]);
		var nb2_col0: f64[:] = new_b2data[0..1];
		var nb2data: f64[:][>..] = []; nb2data.push(nb2_col0);
		let new_b2, nb2ok = matrix::mat(1, 1, copy(nb2data));
		if !nb2ok { print("FAIL: new b2"); failures++; return; }
		w2_b = copy(new_b2);

		e = e + 1;
	}

	print("Training complete.");
	print("");

	let z1_final, z1fok = matrix::dense_forward(w1, X, w1_b);
	if !z1fok { print("FAIL: forward L1 (final)"); failures++; return; }
	matrix::mat_map(z1_final) { x => if x < 0.0 { 0.0 } else { x } };
	let z2_final, z2fok = matrix::dense_forward(w2, z1_final, w2_b);
	if !z2fok { print("FAIL: forward L2 (final)"); failures++; return; }
	functions::mat_sigmoid(z2_final);

	print("--- Predictions AFTER training ---");
	print("Input    Expected  Output");
	print("-------  --------  ------");
	print("0,0      0         ");
	print(z2_final.data[0]);
	print("0,1      1         ");
	print(z2_final.data[1]);
	print("1,0      1         ");
	print(z2_final.data[2]);
	print("1,1      0         ");
	print(z2_final.data[3]);
	print("");

	print("=== EXPECTED vs OUTPUT ===");
	print("");

	print("XOR(0,0)");
	print("expected: 0");
	print("output:   ");
	print(z2_final.data[0]);
	print("");

	print("XOR(0,1)");
	print("expected: 1");
	print("output:   ");
	print(z2_final.data[1]);
	print("");

	print("XOR(1,0)");
	print("expected: 1");
	print("output:   ");
	print(z2_final.data[2]);
	print("");

	print("XOR(1,1)");
	print("expected: 0");
	print("output:   ");
	print(z2_final.data[3]);
	print("");

	print("=== BENCHMARK ===");
	if failures == 0 {
		print("ALL TESTS PASSED");
	} else {
		print("SOME TESTS FAILED (");
		print(failures);
		print(")");
	}
}
