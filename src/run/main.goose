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
	var failures: i64 = 0;
	var results_ok: bool = true;

	// captured output values for end-of-demo benchmark
	var mat_from_0: f64 = 0.0;
	var matmul_0: f64 = 0.0;
	var sum_rows_0: f64 = 0.0;
	var sum_cols_0: f64 = 0.0;
	var mat_row_0: f64 = 0.0;
	var mat_col_0: f64 = 0.0;
	var w_apply_0: f64 = 0.0;
	var b_add_0: f64 = 0.0;
	var dense_fwd_0: f64 = 0.0;
	var dense_bwd_dw_0: f64 = 0.0;
	var mat_relu_1: f64 = 0.0;
	var mat_sigmoid_0: f64 = 0.0;
	var mat_tanh_0: f64 = 0.0;
	var mat_softmax_0: f64 = 0.0;
	var mat_save_load_0: f64 = 0.0;

	// ============================================================
	// STEP 1: mat constructor (2x3 zero-filled)
	// ============================================================
	var empty: f64[:][>..] = [];
	let m, mok = matrix::mat(2, 3, copy(empty));
	if !mok {
		print("Step 1: mat constructor — construct FAILED");
		failures++;
		results_ok = false;
	} else {
		var ok = m.rows == 2 && m.cols != 0;
		for t in 2 * 3 { if m.data[t] != 0.0 { ok = false; } }
		if !ok { print("Step 1: mat constructor — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 2: mat from data [[1,2],[3,4]]
	// ============================================================
	var a1: f64[>..] = []; a1.push(1.0); a1.push(2.0);
	var a2: f64[>..] = []; a2.push(3.0); a2.push(4.0);
	var adata: f64[:][>..] = []; adata.push(a1); adata.push(a2);
	let a, aok = matrix::mat(2, 2, copy(adata));
	if !aok {
		print("Step 2: mat from data — construct FAILED");
		failures++;
		results_ok = false;
	} else {
		var ok = a.rows == 2 && a.cols != 0 &&
			approx(a.data[0], 1.0, 0.0001) &&
			approx(a.data[1], 2.0, 0.0001) &&
			approx(a.data[2], 3.0, 0.0001) &&
			approx(a.data[3], 4.0, 0.0001);
		mat_from_0 = a.data[0];
		if !ok { print("Step 2: mat from data — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 3: matmul [[1,2],[3,4]] x [[5,6],[7,8]]
	// ============================================================
	var bback: f64[>..] = [];
	bback.push(5.0); bback.push(6.0);
	bback.push(7.0); bback.push(8.0);
	var b_row0: f64[:] = bback[0..2];
	var b_row1: f64[:] = bback[2..4];
	var b_data: f64[:][>..] = []; b_data.push(b_row0); b_data.push(b_row1);
	let b, bok = matrix::mat(2, 2, copy(b_data));
	if !bok {
		print("Step 3: matmul — could not create b matrix");
		failures++;
		results_ok = false;
	} else {
		let c, cok = matrix::matmul(a, b);
		if !cok { print("Step 3: matmul — returned false"); failures++; results_ok = false; }
		else {
			var ok = c.rows == 2 && c.cols != 0 &&
				approx(c.data[0], 19.0, 0.0001) &&
				approx(c.data[1], 22.0, 0.0001) &&
				approx(c.data[2], 43.0, 0.0001) &&
				approx(c.data[3], 50.0, 0.0001);
			matmul_0 = c.data[0];
			if !ok { print("Step 3: matmul — FAILED"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// STEP 4: matmul dimension mismatch (should fail)
	// ============================================================
	var badback: f64[>..] = [];
	badback.push(1.0); badback.push(2.0);
	badback.push(3.0); badback.push(4.0);
	badback.push(5.0); badback.push(6.0);
	var bad_row0: f64[:] = badback[0..2];
	var bad_row1: f64[:] = badback[2..4];
	var bad_row2: f64[:] = badback[4..6];
	var bad_data: f64[:][>..] = []; bad_data.push(bad_row0); bad_data.push(bad_row1); bad_data.push(bad_row2);
	let bad, badok = matrix::mat(3, 2, copy(bad_data));
	if !badok { print("Step 4: matmul dimension mismatch — mat bad returned false"); failures++; results_ok = false; }
	else {
		let c2, c2ok = matrix::matmul(a, bad);
		var ok = !c2ok || c2.rows == 0;
		if !ok { print("Step 4: matmul dimension mismatch — should have failed but didn't"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 5: transpose [[1,2],[3,4]] -> [[1,3],[2,4]]
	// ============================================================
	let bt = matrix::transpose(a);
	var ok = bt.rows == 2 && bt.cols != 0 &&
		approx(bt.data[0], 1.0, 0.0001) &&
		approx(bt.data[1], 3.0, 0.0001) &&
		approx(bt.data[2], 2.0, 0.0001) &&
		approx(bt.data[3], 4.0, 0.0001);
	if !ok { print("Step 5: transpose — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 6: mat_add [[1,2],[3,4]] + [[5,6],[7,8]]
	// ============================================================
	var acopy = matrix::mat_cpy(a);
	let d, dok = matrix::mat_add(acopy, b);
	if !dok { print("Step 6: mat_add — returned false"); failures++; results_ok = false; }
	else {
		var ok = d.rows == 2 && d.cols != 0 &&
			approx(d.data[0], 6.0, 0.0001) &&
			approx(d.data[1], 8.0, 0.0001) &&
			approx(d.data[2], 10.0, 0.0001) &&
			approx(d.data[3], 12.0, 0.0001);
		if !ok { print("Step 6: mat_add — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 7: mat_mul (Hadamard) [[1,2],[3,4]] * [[5,6],[7,8]]
	// ============================================================
	let e, eok = matrix::mat_mul(acopy, b);
	if !eok { print("Step 7: mat_mul — returned false"); failures++; results_ok = false; }
	else {
		var ok = e.rows == 2 && e.cols != 0 &&
			approx(e.data[0], 5.0, 0.0001) &&
			approx(e.data[1], 12.0, 0.0001) &&
			approx(e.data[2], 21.0, 0.0001) &&
			approx(e.data[3], 32.0, 0.0001);
		if !ok { print("Step 7: mat_mul — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 8: mat_div [[2,4],[6,8]] / [[2,2],[2,2]]
	// ============================================================
	var dbck: f64[>..] = [];
	dbck.push(2.0); dbck.push(4.0);
	dbck.push(6.0); dbck.push(8.0);
	var db_row0: f64[:] = dbck[0..2];
	var db_row1: f64[:] = dbck[2..4];
	var bz_data: f64[:][>..] = []; bz_data.push(db_row0); bz_data.push(db_row1);
	let dmat, bok2 = matrix::mat(2, 2, copy(bz_data));
	var bvck: f64[>..] = [];
	bvck.push(2.0); bvck.push(2.0);
	bvck.push(2.0); bvck.push(2.0);
	var bv_row0: f64[:] = bvck[0..2];
	var bv_row1: f64[:] = bvck[2..4];
	var bv_data: f64[:][>..] = []; bv_data.push(bv_row0); bv_data.push(bv_row1);
	let bmat, bok3 = matrix::mat(2, 2, copy(bv_data));
	if !bok2 || !bok3 { print("Step 8: mat_div — matrix creation failed"); failures++; results_ok = false; }
	else {
		let f, fok = matrix::mat_div(dmat, bmat);
		if !fok { print("Step 8: mat_div — returned false"); failures++; results_ok = false; }
		else {
			var ok = f.rows == 2 && f.cols != 0 &&
				approx(f.data[0], 1.0, 0.0001) &&
				approx(f.data[1], 2.0, 0.0001) &&
				approx(f.data[2], 3.0, 0.0001) &&
				approx(f.data[3], 4.0, 0.0001);
			if !ok { print("Step 8: mat_div — FAILED"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// STEP 9: mat_div by zero (should fail)
	// ============================================================
	var zbck: f64[>..] = [];
	zbck.push(0.0); zbck.push(1.0);
	zbck.push(1.0); zbck.push(1.0);
	var zb_row0: f64[:] = zbck[0..2];
	var zb_row1: f64[:] = zbck[2..4];
	var zb_data: f64[:][>..] = []; zb_data.push(zb_row0); zb_data.push(zb_row1);
	let zbm, zbok = matrix::mat(2, 2, copy(zb_data));
	if !zbok { print("Step 9: mat_div by zero — mat zbm creation failed"); failures++; results_ok = false; }
	else {
		let g, gok = matrix::mat_div(dmat, zbm);
		var ok = !gok;
		if !ok { print("Step 9: mat_div by zero — should have failed but didn't"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 10: mat_sum_rows [[1,2],[3,4]] -> [3,7]
	// ============================================================
	let sums = matrix::mat_sum_rows(a);
	var ok = sums.len == 2 && approx(sums[0], 3.0, 0.0001) && approx(sums[1], 7.0, 0.0001);
	sum_rows_0 = sums[0];
	if !ok { print("Step 10: mat_sum_rows — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 11: mat_sum_cols [[1,2],[3,4]] -> [4,6]
	// ============================================================
	let sumc = matrix::mat_sum_cols(a);
	var ok = sumc.len == 2 && approx(sumc[0], 4.0, 0.0001) && approx(sumc[1], 6.0, 0.0001);
	sum_cols_0 = sumc[0];
	if !ok { print("Step 11: mat_sum_cols — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 12: mat_row [[1,2],[3,4]] row 0 -> [1,2]
	// ============================================================
	let r, rok = matrix::mat_row(a, 0);
	if !rok { print("Step 12: mat_row — returned false"); failures++; results_ok = false; }
	else {
		var ok = r.len == 2 && approx(r[0], 1.0, 0.0001) && approx(r[1], 2.0, 0.0001);
		mat_row_0 = r[0];
		if !ok { print("Step 12: mat_row — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 13: mat_col [[1,2],[3,4]] col 0 -> [1,3]
	// ============================================================
	let ccol, ccolok = matrix::mat_col(a, 0);
	if !ccolok { print("Step 13: mat_col — returned false"); failures++; results_ok = false; }
	else {
		var ok = ccol.len == 2 && approx(ccol[0], 1.0, 0.0001) && approx(ccol[1], 3.0, 0.0001);
		mat_col_0 = ccol[0];
		if !ok { print("Step 13: mat_col — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 14: mat_cpy [[1,2],[3,4]]
	// ============================================================
	let cp = matrix::mat_cpy(a);
	var ok = cp.rows == 2 && cp.cols != 0 &&
		approx(cp.data[0], 1.0, 0.0001) &&
		approx(cp.data[1], 2.0, 0.0001) &&
		approx(cp.data[2], 3.0, 0.0001) &&
		approx(cp.data[3], 4.0, 0.0001);
	if !ok { print("Step 14: mat_cpy — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 15: mat_map [[1,2],[3,4]] * 2
	// ============================================================
	matrix::mat_map(a) { x => x * 2.0 };
	var ok = approx(a.data[0], 2.0, 0.0001) &&
		approx(a.data[1], 4.0, 0.0001) &&
		approx(a.data[2], 6.0, 0.0001) &&
		approx(a.data[3], 8.0, 0.0001);
	if !ok { print("Step 15: mat_map — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 16: weights w_init reproducibility
	// ============================================================
	let w = weights::w_init(2, 3, 7);
	let w2 = weights::w_init(2, 3, 7);
	var wsame = true;
	for t in 6 { if w.data[t] != w2.data[t] { wsame = false; } }
	if !wsame { print("Step 16: w_init reproducibility — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 17: weights w_set (0,0) = 99
	// ============================================================
	let wok = weights::w_set(w, 0, 0, [99.0]);
	if !wok { print("Step 17: w_set — returned false"); failures++; results_ok = false; }
	else {
		var ok = approx(w.data[0], 99.0, 0.0001);
		if !ok { print("Step 17: w_set — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 18: weights w_apply [[1,2,3],[4,5,6]] x [[1],[2],[3]]
	// ============================================================
	var wfback: f64[>..] = [];
	wfback.push(1.0); wfback.push(2.0); wfback.push(3.0);
	wfback.push(4.0); wfback.push(5.0); wfback.push(6.0);
	var wf_row0: f64[:] = wfback[0..3];
	var wf_row1: f64[:] = wfback[3..6];
	var wf_data: f64[:][>..] = []; wf_data.push(wf_row0); wf_data.push(wf_row1);
	let wf, wfok = matrix::mat(2, 3, copy(wf_data));
	if !wfok { print("Step 18: w_apply — could not create w matrix"); failures++; results_ok = false; }
	else {
		var xinback: f64[>..] = [];
		xinback.push(1.0); xinback.push(2.0); xinback.push(3.0);
		var xin_row0: f64[:] = xinback[0..1];
		var xin_row1: f64[:] = xinback[1..2];
		var xin_row2: f64[:] = xinback[2..3];
		var xin_data: f64[:][>..] = []; xin_data.push(xin_row0); xin_data.push(xin_row1); xin_data.push(xin_row2);
		let xin, xinok = matrix::mat(3, 1, copy(xin_data));
		if !xinok { print("Step 18: w_apply — could not create xin matrix"); failures++; results_ok = false; }
		else {
			let wc, wcok = weights::w_apply(wf, xin);
			if !wcok { print("Step 18: w_apply — returned false"); failures++; results_ok = false; }
			else {
				var ok = wc.rows == 2 && wc.cols != 0 &&
					approx(wc.data[0], 14.0, 0.0001) &&
					approx(wc.data[1], 32.0, 0.0001);
				w_apply_0 = wc.data[0];
				if !ok { print("Step 18: w_apply — FAILED"); failures++; results_ok = false; }
			}
		}
	}

	// ============================================================
	// STEP 19: weights w_apply dimension mismatch (should fail)
	// ============================================================
	let wbad = weights::w_init(3, 2, 1);
	var wbback: f64[>..] = [];
	wbback.push(1.0); wbback.push(2.0); wbback.push(3.0);
	wbback.push(4.0); wbback.push(5.0); wbback.push(6.0);
	wbback.push(7.0); wbback.push(8.0); wbback.push(9.0);
	var wb_row0: f64[:] = wbback[0..3];
	var wb_row1: f64[:] = wbback[3..6];
	var wb_row2: f64[:] = wbback[6..9];
	var wbad_data: f64[:][>..] = []; wbad_data.push(wb_row0); wbad_data.push(wb_row1); wbad_data.push(wb_row2);
	let wbad_input, winpok = matrix::mat(3, 3, copy(wbad_data));
	if !winpok { print("Step 19: w_apply dimension mismatch — mat wbad_input returned false"); failures++; results_ok = false; }
	else {
		let wc2, wc2ok = weights::w_apply(wbad, wbad_input);
		var ok = !wc2ok;
		if !ok { print("Step 19: w_apply dimension mismatch — should have failed but didn't"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 20: bias b_init(3) zero 1x3
	// ============================================================
	let bb = bias::b_init(3);
	var ok = bb.rows == 1 && bb.cols == 3 &&
		bb.data[0] == 0.0 && bb.data[1] == 0.0 && bb.data[2] == 0.0;
	if !ok { print("Step 20: b_init — FAILED"); failures++; results_ok = false; }

	// ============================================================
	// STEP 21: bias b_set [[0,0,0]] row 0 col 1 = 5
	// ============================================================
	let bok = bias::b_set(bb, 1, [5.0]);
	if !bok { print("Step 21: b_set — returned false"); failures++; results_ok = false; }
	else {
		var ok = approx(bb.data[1], 5.0, 0.0001);
		if !ok { print("Step 21: b_set — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 22: bias b_add [[1,2],[3,4]] + [[10,20]] -> [[11,22],[13,24]]
	// ============================================================
	var bbck: f64[>..] = [];
	bbck.push(10.0); bbck.push(20.0);
	var bb_row0: f64[:] = bbck[0..2];
	var bb_data: f64[:][>..] = []; bb_data.push(bb_row0);
	var bb2, bb2ok = matrix::mat(1, 2, copy(bb_data));
	if !bb2ok { print("Step 22: b_add — mat bb2 creation failed"); failures++; results_ok = false; }
	else {
		let ba, baok = bias::b_add(acopy, bb2);
		if !baok { print("Step 22: b_add — returned false"); failures++; results_ok = false; }
		else {
			var ok = ba.rows == 2 && ba.cols != 0 &&
				approx(ba.data[0], 11.0, 0.0001) &&
				approx(ba.data[1], 22.0, 0.0001) &&
				approx(ba.data[2], 13.0, 0.0001) &&
				approx(ba.data[3], 24.0, 0.0001);
			b_add_0 = ba.data[0];
			if !ok { print("Step 22: b_add — FAILED"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// STEP 23: bias b_add dimension mismatch (should fail)
	// ============================================================
	let bbad = bias::b_init(5);
	let ba2, ba2ok = bias::b_add(a, bbad);
	var ok = !ba2ok;
	if !ok { print("Step 23: b_add dimension mismatch — should have failed but didn't"); failures++; results_ok = false; }

	// ============================================================
	// STEP 24: dense forward (two layers) [[1,2],[3,4]] -> [16]
	// ============================================================
	var w1back: f64[>..] = [];
	w1back.push(1.0); w1back.push(2.0);
	w1back.push(3.0); w1back.push(4.0);
	var w1_row0: f64[:] = w1back[0..2];
	var w1_row1: f64[:] = w1back[2..4];
	var w1_data: f64[:][>..] = []; w1_data.push(w1_row0); w1_data.push(w1_row1);
	let w1, w1ok = matrix::mat(2, 2, copy(w1_data));
	if !w1ok { print("Step 24: dense forward — mat w1 returned false"); failures++; results_ok = false; }
	var b1back: f64[>..] = [];
	b1back.push(0.0);
	var b1_row0: f64[:] = b1back[0..1];
	var b1_data: f64[:][>..] = []; b1_data.push(b1_row0);
	let b1, b1ok = matrix::mat(1, 1, copy(b1_data));
	if !b1ok { print("Step 24: dense forward — mat b1 returned false"); failures++; results_ok = false; }
	var w2back: f64[>..] = [];
	w2back.push(1.0); w2back.push(1.0);
	var w2_row0: f64[:] = w2back[0..2];
	var w2_data: f64[:][>..] = []; w2_data.push(w2_row0);
	let w2, w2ok = matrix::mat(1, 2, copy(w2_data));
	if !w2ok { print("Step 24: dense forward — mat w2 returned false"); failures++; results_ok = false; }
	var b2back: f64[>..] = [];
	b2back.push(0.0);
	var b2_row0: f64[:] = b2back[0..1];
	var b2_data: f64[:][>..] = []; b2_data.push(b2_row0);
	let b2, b2ok = matrix::mat(1, 1, copy(b2_data));
	if !b2ok { print("Step 24: dense forward — mat b2 returned false"); failures++; results_ok = false; }
	var xin_flat: f64[>..] = [];
	xin_flat.push(1.0);
	xin_flat.push(2.0);
	var xin_row0: f64[:] = xin_flat[0..1];
	var xin_row1: f64[:] = xin_flat[1..2];
	var xin_data2: f64[:][>..] = []; xin_data2.push(xin_row0); xin_data2.push(xin_row1);
	let xin2, xin2ok = matrix::mat(2, 1, copy(xin_data2));
	if !xin2ok { print("Step 24: dense forward — mat xin2 returned false"); failures++; results_ok = false; }
	let z1, z1ok = matrix::dense_forward(w1, xin2, b1);
	if !z1ok { print("Step 24: dense forward — layer 1 failed"); failures++; results_ok = false; }
	else {
		matrix::mat_map(z1) { x => if x < 0.0 { 0.0 } else { x } };
		let z2, z2ok = matrix::dense_forward(w2, z1, b2);
		if !z2ok { print("Step 24: dense forward — layer 2 failed"); failures++; results_ok = false; }
		else {
			matrix::mat_map(z2) { x => if x < 0.0 { 0.0 } else { x } };
			var ok = z2.rows == 1 && z2.cols != 0 && approx(z2.data[0], 16.0, 0.0001);
			dense_fwd_0 = z2.data[0];
			if !ok { print("Step 24: dense forward — FAILED"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// STEP 25: dense backward
	// ============================================================
	var twback: f64[>..] = [];
	twback.push(1.0); twback.push(2.0); twback.push(3.0);
	twback.push(4.0); twback.push(5.0); twback.push(6.0);
	var tw_row0: f64[:] = twback[0..3];
	var tw_row1: f64[:] = twback[3..6];
	var tw_data: f64[:][>..] = []; tw_data.push(tw_row0); tw_data.push(tw_row1);
	let twk, twkok = matrix::mat(2, 3, copy(tw_data));
	if !twkok { print("Step 25: dense_backward — mat twk returned false"); failures++; results_ok = false; }
	var xback2: f64[>..] = [];
	xback2.push(7.0); xback2.push(8.0);
	xback2.push(9.0); xback2.push(10.0);
	xback2.push(11.0); xback2.push(12.0);
	var x_row0: f64[:] = xback2[0..2];
	var x_row1: f64[:] = xback2[2..4];
	var x_row2: f64[:] = xback2[4..6];
	var x_data2: f64[:][>..] = []; x_data2.push(x_row0); x_data2.push(x_row1); x_data2.push(x_row2);
	let x2, x2ok = matrix::mat(3, 2, copy(x_data2));
	if !x2ok { print("Step 25: dense_backward — mat x2 returned false"); failures++; results_ok = false; }
	var dzback2: f64[>..] = [];
	dzback2.push(1.0); dzback2.push(2.0);
	dzback2.push(3.0); dzback2.push(4.0);
	var dz_row0: f64[:] = dzback2[0..2];
	var dz_row1: f64[:] = dzback2[2..4];
	var dz_data2: f64[:][>..] = []; dz_data2.push(dz_row0); dz_data2.push(dz_row1);
	let dz2, dz2ok = matrix::mat(2, 2, copy(dz_data2));
	if !dz2ok { print("Step 25: dense_backward — mat dz2 returned false"); failures++; results_ok = false; }
	let dw, db, dx, dok = matrix::dense_backward(twk, x2, dz2);
	if !dok { print("Step 25: dense_backward — returned false"); failures++; results_ok = false; }
	else {
		var ok = dw.rows == 2 && dw.cols != 0 &&
			approx(dw.data[0], 23.0, 0.0001) &&
			approx(dw.data[1], 29.0, 0.0001) &&
			approx(dw.data[2], 35.0, 0.0001) &&
			approx(dw.data[3], 53.0, 0.0001) &&
			approx(dw.data[4], 67.0, 0.0001) &&
			approx(dw.data[5], 81.0, 0.0001) &&
			db.len == 2 &&
			approx(db[0], 4.0, 0.0001) &&
			approx(db[1], 6.0, 0.0001) &&
			dx.rows == 3 && dx.cols != 0 &&
			approx(dx.data[0], 13.0, 0.0001) &&
			approx(dx.data[1], 18.0, 0.0001) &&
			approx(dx.data[2], 17.0, 0.0001) &&
			approx(dx.data[3], 24.0, 0.0001) &&
			approx(dx.data[4], 21.0, 0.0001) &&
			approx(dx.data[5], 30.0, 0.0001);
		dense_bwd_dw_0 = dw.data[0];
		if !ok { print("Step 25: dense_backward — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 26: mat_relu [[-1,2],[-3,4]] -> [[0,2],[0,4]]
	// ============================================================
	var rrback: f64[>..] = [];
	rrback.push(-1.0); rrback.push(2.0);
	rrback.push(-3.0); rrback.push(4.0);
	var rr_row0: f64[:] = rrback[0..2];
	var rr_row1: f64[:] = rrback[2..4];
	var rrdata: f64[:][>..] = []; rrdata.push(rr_row0); rrdata.push(rr_row1);
	let rm, rmok = matrix::mat(2, 2, copy(rrdata));
	if !rmok { print("Step 26: mat_relu — mat rm returned false"); failures++; results_ok = false; }
	else {
		functions::mat_relu(rm);
		var ok = approx(rm.data[0], 0.0, 0.0001) &&
			approx(rm.data[1], 2.0, 0.0001) &&
			approx(rm.data[2], 0.0, 0.0001) &&
			approx(rm.data[3], 4.0, 0.0001);
		mat_relu_1 = rm.data[1];
		if !ok { print("Step 26: mat_relu — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 27: mat_sigmoid [[0]] -> [0.5]
	// ============================================================
	var srback: f64[>..] = [];
	srback.push(0.0);
	var sr_row0: f64[:] = srback[0..1];
	var srdata: f64[:][>..] = []; srdata.push(sr_row0);
	let sm, smok = matrix::mat(1, 1, copy(srdata));
	if !smok { print("Step 27: mat_sigmoid — mat sm returned false"); failures++; results_ok = false; }
	else {
		functions::mat_sigmoid(sm);
		var ok = approx(sm.data[0], 0.5, 0.0001);
		mat_sigmoid_0 = sm.data[0];
		if !ok { print("Step 27: mat_sigmoid — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 28: mat_tanh [[0]] -> [0]
	// ============================================================
	var trback: f64[>..] = [];
	trback.push(0.0);
	var tr_row0: f64[:] = trback[0..1];
	var trdata: f64[:][>..] = []; trdata.push(tr_row0);
	let tm, ttok = matrix::mat(1, 1, copy(trdata));
	if !ttok { print("Step 28: mat_tanh — mat tm returned false"); failures++; results_ok = false; }
	else {
		functions::mat_tanh(tm);
		var ok = approx(tm.data[0], 0.0, 0.0001);
		mat_tanh_0 = tm.data[0];
		if !ok { print("Step 28: mat_tanh — FAILED"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 29: mat_softmax_rows [[0,0]] -> [0.5,0.5]
	// ============================================================
	var smback: f64[>..] = [];
	smback.push(0.0); smback.push(0.0);
	var sm_row0: f64[:] = smback[0..2];
	var smdata: f64[:][>..] = []; smdata.push(sm_row0);
	let smm, smmok = matrix::mat(1, 2, copy(smdata));
	if !smmok { print("Step 29: mat_softmax_rows — mat smm returned false"); failures++; results_ok = false; }
	else {
		let sok = functions::mat_softmax_rows(smm);
		if !sok { print("Step 29: mat_softmax_rows — returned false"); failures++; results_ok = false; }
		else {
			var ok = approx(smm.data[0], 0.5, 0.0001) && approx(smm.data[1], 0.5, 0.0001);
			mat_softmax_0 = smm.data[0];
			if !ok { print("Step 29: mat_softmax_rows — FAILED"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// STEP 30: softmax [[-1000,-1000]] underflow (should fail)
	// ============================================================
	var sfback: f64[>..] = [];
	sfback.push(-1000.0); sfback.push(-1000.0);
	var sf_row0: f64[:] = sfback[0..2];
	var sfdata: f64[:][>..] = []; sfdata.push(sf_row0);
	let smf, smfok = matrix::mat(1, 2, copy(sfdata));
	if !smfok { print("Step 30: softmax underflow — mat smf returned false"); failures++; results_ok = false; }
	else {
		let sfok = functions::mat_softmax_rows(smf);
		var ok = !sfok;
		if !ok { print("Step 30: softmax underflow — should have failed but didn't"); failures++; results_ok = false; }
	}

	// ============================================================
	// STEP 31: mat_save / mat_load [[1,2],[3,4]]
	// ============================================================
	let save_mat2, smok2 = matrix::mat(2, 2, copy(adata));
	if !smok2 { print("Step 31: mat_save/mat_load — mat save_mat returned false"); failures++; results_ok = false; }
	else {
		let save_ok = matrix::mat_save(save_mat2, "/tmp/test_matrix.txt");
		if !save_ok { print("Step 31: mat_save/mat_load — mat_save returned false"); failures++; results_ok = false; }
		else {
			let load_mat, lok = matrix::mat_load("/tmp/test_matrix.txt");
			if !lok { print("Step 31: mat_save/mat_load — mat_load returned false"); failures++; results_ok = false; }
			else {
				var ok = load_mat.rows == 2 && load_mat.cols != 0 &&
					approx(load_mat.data[0], 1.0, 0.0001) &&
					approx(load_mat.data[1], 2.0, 0.0001) &&
					approx(load_mat.data[2], 3.0, 0.0001) &&
					approx(load_mat.data[3], 4.0, 0.0001);
				mat_save_load_0 = load_mat.data[0];
				if !ok { print("Step 31: mat_save/mat_load — FAILED"); failures++; results_ok = false; }
			}
		}
	}

	// ============================================================
	// STEP 32: dense_train_step [[1,2],[3,4]] x [[7],[8]] -> [[15],[31]]
	// ============================================================
	let twback: f64[>..] = [];
	twback.push(1.0); twback.push(2.0);
	twback.push(3.0); twback.push(4.0);
	var tw_row0: f64[:] = twback[0..2];
	var tw_row1: f64[:] = twback[2..4];
	var tw_data: f64[:][>..] = []; tw_data.push(tw_row0); tw_data.push(tw_row1);
	let twk, twkok = matrix::mat(2, 2, copy(tw_data));
	if !twkok { print("Step 32: dense_train_step — mat twk returned false"); failures++; results_ok = false; }
	var txback2: f64[>..] = [];
	txback2.push(7.0); txback2.push(8.0);
	var tx_row0: f64[:] = txback2[0..1];
	var tx_row1: f64[:] = txback2[1..2];
	var tx_data2: f64[:][>..] = []; tx_data2.push(tx_row0); tx_data2.push(tx_row1);
	let tx2, tx2ok = matrix::mat(2, 1, copy(tx_data2));
	if !tx2ok { print("Step 32: dense_train_step — mat tx2 returned false"); failures++; results_ok = false; }
	var tyback2: f64[>..] = [];
	tyback2.push(15.0); tyback2.push(31.0);
	var ty_row0: f64[:] = tyback2[0..1];
	var ty_row1: f64[:] = tyback2[1..2];
	var ty_data2: f64[:][>..] = []; ty_data2.push(ty_row0); ty_data2.push(ty_row1);
	let ty2, ty2ok = matrix::mat(2, 1, copy(ty_data2));
	if !ty2ok { print("Step 32: dense_train_step — mat ty2 returned false"); failures++; results_ok = false; }
	var tbback: f64[>..] = [];
	tbback.push(0.0);
	var tb_row0: f64[:] = tbback[0..1];
	var tb_data: f64[:][>..] = []; tb_data.push(tb_row0);
	let tb, tbok = matrix::mat(1, 1, copy(tb_data));
	if !tbok { print("Step 32: dense_train_step — mat tb returned false"); failures++; results_ok = false; }
	let loss1, lok1 = matrix::dense_train_step(twk, tb, tx2, ty2, 0.01);
	if !lok1 { print("Step 32: dense_train_step — returned false"); failures++; results_ok = false; }
	else {
		let loss2, lok2 = matrix::dense_train_step(twk, tb, tx2, ty2, 0.01);
		if !lok2 { print("Step 32: dense_train_step — second call returned false"); failures++; results_ok = false; }
		else {
			var ok = loss2 < loss1;
			if !ok { print("Step 32: dense_train_step — loss did not decrease"); failures++; results_ok = false; }
		}
	}

	// ============================================================
	// EXPECTED vs OUTPUT (at the end)
	// ============================================================
	print("");
	print("=== EXPECTED vs OUTPUT ===");
	print("");

	print("mat from data [0,0]");
	print("expected: 1");
	print("output:   ");
	print(mat_from_0);
	print("");

	print("matmul [0,0]");
	print("expected: 19");
	print("output:   ");
	print(matmul_0);
	print("");

	print("mat_sum_rows [0]");
	print("expected: 3");
	print("output:   ");
	print(sum_rows_0);
	print("");

	print("mat_sum_cols [0]");
	print("expected: 4");
	print("output:   ");
	print(sum_cols_0);
	print("");

	print("mat_row [0]");
	print("expected: 1");
	print("output:   ");
	print(mat_row_0);
	print("");

	print("mat_col [0]");
	print("expected: 1");
	print("output:   ");
	print(mat_col_0);
	print("");

	print("w_apply [0]");
	print("expected: 14");
	print("output:   ");
	print(w_apply_0);
	print("");

	print("b_add [0]");
	print("expected: 11");
	print("output:   ");
	print(b_add_0);
	print("");

	print("dense forward [0]");
	print("expected: 16");
	print("output:   ");
	print(dense_fwd_0);
	print("");

	print("dense_backward dw[0]");
	print("expected: 23");
	print("output:   ");
	print(dense_bwd_dw_0);
	print("");

	print("mat_relu [1]");
	print("expected: 2");
	print("output:   ");
	print(mat_relu_1);
	print("");

	print("mat_sigmoid [0]");
	print("expected: 0.5");
	print("output:   ");
	print(mat_sigmoid_0);
	print("");

	print("mat_tanh [0]");
	print("expected: 0");
	print("output:   ");
	print(mat_tanh_0);
	print("");

	print("mat_softmax_rows [0]");
	print("expected: 0.5");
	print("output:   ");
	print(mat_softmax_0);
	print("");

	print("mat_save/load [0]");
	print("expected: 1");
	print("output:   ");
	print(mat_save_load_0);
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
