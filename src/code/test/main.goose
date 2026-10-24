// main.goose — runtime tests for matrix, weights, bias, and functions modules.
// Run with: goose build main.goose && ./main

import matrix from "../include/matrix";
import weights from "../include/weights";
import bias from "../include/bias";
import functions from "../include/functions";

fn approx(a: f64, b: f64, eps: f64) -> bool {
	var d = a - b;
	if d < 0.0 { d = -d; }
	return d < eps;
}

fn main() {
	var failures: i64 = 0;

	// --- mat constructor (zero-filled from empty data) ---
	var empty: f64[:][>..] = [];
	let m, mok = matrix::mat(2, 3, copy(empty));
	if !mok { failures++; print("FAIL: mat constructor returned false"); }
	if m.rows != 2 || m.cols != 3 { failures++; print("FAIL: mat dims"); }
	for t in 2 * 3 {
		if m.data[t] != 0.0 { failures++; print("FAIL: mat not zero"); break; }
	}

	// --- mat from data ---
	var backing: f64[>..] = [];
	backing.push(1.0); backing.push(2.0); backing.push(3.0);
	backing.push(4.0); backing.push(5.0); backing.push(6.0);
	var row0: f64[:] = backing[0..3];
	var row1: f64[:] = backing[3..6];
	var d2: f64[:][>..] = []; d2.push(row0); d2.push(row1);
	let m2, m2ok = matrix::mat(2, 3, copy(d2));
	if !m2ok { failures++; print("FAIL: mat from data returned false"); }
	if !approx(m2.data[0], 1.0, 0.0001) { failures++; print("FAIL: mat data[0]"); }
	if !approx(m2.data[5], 6.0, 0.0001) { failures++; print("FAIL: mat data[5]"); }

	// --- mat from data wrong rows ---
	var wrong_backing: f64[>..] = [];
	wrong_backing.push(1.0); wrong_backing.push(2.0); wrong_backing.push(3.0);
	var wrong_row: f64[:] = wrong_backing[0..3];
	var wrong_rows: f64[:][>..] = []; wrong_rows.push(wrong_row);
	let m3, m3ok = matrix::mat(2, 3, copy(wrong_rows));
	if m3ok { failures++; print("FAIL: mat wrong rows should be false"); }

	// --- mat from data wrong cols ---
	var wc_backing: f64[>..] = [];
	wc_backing.push(1.0); wc_backing.push(2.0);
	wc_backing.push(3.0); wc_backing.push(4.0); wc_backing.push(5.0);
	var wc_row1: f64[:] = wc_backing[0..2];
	var wc_row2: f64[:] = wc_backing[2..5];
	var wc_data: f64[:][>..] = []; wc_data.push(wc_row1); wc_data.push(wc_row2);
	let m4, m4ok = matrix::mat(2, 2, copy(wc_data));
	if m4ok { failures++; print("FAIL: mat wrong cols should be false"); }

	// --- mat_gen reproducibility ---
	let g1 = matrix::mat_gen(2, 2, 42);
	let g2 = matrix::mat_gen(2, 2, 42);
	let g3 = matrix::mat_gen(2, 2, 99);
	var same = true;
	for t in 4 { if g1.data[t] != g2.data[t] { same = false; } }
	if !same { failures++; print("FAIL: mat_gen not reproducible"); }
	var diff = false;
	for t in 4 { if g1.data[t] != g3.data[t] { diff = true; } }
	if !diff { failures++; print("FAIL: mat_gen different seeds should differ"); }

	// --- matmul with known values ---
	// a = [[1,2],[3,4]] (2x2), b = [[5,6],[7,8]] (2x2)
	// expected = [[19,22],[43,50]]
	var aback: f64[>..] = [];
	aback.push(1.0); aback.push(2.0);
	aback.push(3.0); aback.push(4.0);
	var a_row0: f64[:] = aback[0..2];
	var a_row1: f64[:] = aback[2..4];
	var adata: f64[:][>..] = []; adata.push(a_row0); adata.push(a_row1);
	let a, aok = matrix::mat(2, 2, copy(adata));
	if !aok { failures++; print("FAIL: mat a returned false"); }
	var bback: f64[>..] = [];
	bback.push(5.0); bback.push(6.0);
	bback.push(7.0); bback.push(8.0);
	var b_row0: f64[:] = bback[0..2];
	var b_row1: f64[:] = bback[2..4];
	var bdata: f64[:][>..] = []; bdata.push(b_row0); bdata.push(b_row1);
	let b, bok = matrix::mat(2, 2, copy(bdata));
	if !bok { failures++; print("FAIL: mat b returned false"); }
	let c, cok = matrix::matmul(a, b);
	if !cok { failures++; print("FAIL: matmul returned false"); }
	if c.rows != 2 || c.cols != 2 { failures++; print("FAIL: matmul dims"); }
	if !approx(c.data[0], 19.0, 0.0001) { failures++; print("FAIL: matmul[0]"); }
	if !approx(c.data[1], 22.0, 0.0001) { failures++; print("FAIL: matmul[1]"); }
	if !approx(c.data[2], 43.0, 0.0001) { failures++; print("FAIL: matmul[2]"); }
	if !approx(c.data[3], 50.0, 0.0001) { failures++; print("FAIL: matmul[3]"); }

	// --- matmul dimension mismatch ---
	var badback: f64[>..] = [];
	badback.push(1.0); badback.push(2.0);
	badback.push(3.0); badback.push(4.0);
	badback.push(5.0); badback.push(6.0);
	var bad_row0: f64[:] = badback[0..2];
	var bad_row1: f64[:] = badback[2..4];
	var bad_row2: f64[:] = badback[4..6];
	var bad_data: f64[:][>..] = []; bad_data.push(bad_row0); bad_data.push(bad_row1); bad_data.push(bad_row2);
	let bad, badok = matrix::mat(3, 2, copy(bad_data));
	if !badok { failures++; print("FAIL: mat bad returned false"); }
	let c2, c2ok = matrix::matmul(a, bad);
	if c2ok { failures++; print("FAIL: matmul mismatch should be false"); }
	if c2.rows != 0 { failures++; print("FAIL: matmul mismatch zero rows"); }

	// --- mat_add ---
	let d_add, dok = matrix::mat_add(a, b);
	if !dok { failures++; print("FAIL: mat_add returned false"); }
	if !approx(d_add.data[0], 6.0, 0.0001) { failures++; print("FAIL: mat_add[0]"); }

	// --- mat_mul (Hadamard) ---
	let e, eok = matrix::mat_mul(a, b);
	if !eok { failures++; print("FAIL: mat_mul returned false"); }
	if !approx(e.data[0], 5.0, 0.0001) { failures++; print("FAIL: mat_mul[0]"); }

	// --- mat_div ---
	let f, fok = matrix::mat_div(a, b);
	if !fok { failures++; print("FAIL: mat_div returned false"); }
	if !approx(f.data[0], 0.2, 0.0001) { failures++; print("FAIL: mat_div[0]"); }

	// --- mat_div by zero ---
	var bzback: f64[>..] = [];
	bzback.push(0.0); bzback.push(1.0);
	bzback.push(1.0); bzback.push(1.0);
	var bz_row0: f64[:] = bzback[0..2];
	var bz_row1: f64[:] = bzback[2..4];
	var bzdata: f64[:][>..] = []; bzdata.push(bz_row0); bzdata.push(bz_row1);
	let bz, bzok = matrix::mat(2, 2, copy(bzdata));
	if !bzok { failures++; print("FAIL: mat bz returned false"); }
	let g, gok = matrix::mat_div(a, bz);
	if gok { failures++; print("FAIL: mat_div by zero should be false"); }

	// --- mat_sum_rows ---
	let sums = matrix::mat_sum_rows(a);
	if sums[0] != 3.0 { failures++; print("FAIL: sum_rows[0]"); }
	if sums[1] != 7.0 { failures++; print("FAIL: sum_rows[1]"); }

	// --- mat_sum_cols ---
	let sumc = matrix::mat_sum_cols(a);
	if sumc[0] != 4.0 { failures++; print("FAIL: sum_cols[0]"); }
	if sumc[1] != 6.0 { failures++; print("FAIL: sum_cols[1]"); }

	// --- mat_row ---
	let r, rok = matrix::mat_row(a, 0);
	if !rok { failures++; print("FAIL: mat_row returned false"); }
	if r[0] != 1.0 || r[1] != 2.0 { failures++; print("FAIL: mat_row data"); }
	let r2, r2ok = matrix::mat_row(a, 5);
	if r2ok { failures++; print("FAIL: mat_row bad index should be false"); }

	// --- mat_col ---
	let ccol, ccolok = matrix::mat_col(a, 0);
	if !ccolok { failures++; print("FAIL: mat_col returned false"); }
	if ccol[0] != 1.0 || ccol[1] != 3.0 { failures++; print("FAIL: mat_col data"); }

	// --- mat_reshape ---
	let reshaped, rokok = matrix::mat_reshape(a, 1, 4);
	if !rokok { failures++; print("FAIL: mat_reshape returned false"); }
	if reshaped.rows != 1 || reshaped.cols != 4 { failures++; print("FAIL: mat_reshape dims"); }
	let badreshape, brkok = matrix::mat_reshape(a, 3, 3);
	if brkok { failures++; print("FAIL: mat_reshape bad size should be false"); }

	// --- mat_concat_rows ---
	let cr, crok = matrix::mat_concat_rows(a, b);
	if !crok { failures++; print("FAIL: mat_concat_rows returned false"); }
	if cr.rows != 4 || cr.cols != 2 { failures++; print("FAIL: concat_rows dims"); }

	// --- mat_concat_cols ---
	let cc, cck = matrix::mat_concat_cols(a, b);
	if !cck { failures++; print("FAIL: mat_concat_cols returned false"); }
	if cc.rows != 2 || cc.cols != 4 { failures++; print("FAIL: concat_cols dims"); }

	// --- transpose ---
	let bt = matrix::transpose(a);
	if bt.rows != 2 || bt.cols != 2 { failures++; print("FAIL: transpose dims"); }
	if !approx(bt.data[0], 1.0, 0.0001) { failures++; print("FAIL: transpose[0]"); }
	if !approx(bt.data[1], 3.0, 0.0001) { failures++; print("FAIL: transpose[1]"); }

	// --- mat_cpy ---
	let cp = matrix::mat_cpy(a);
	if cp.rows != 2 || cp.cols != 2 { failures++; print("FAIL: cpy dims"); }
	if !approx(cp.data[0], 1.0, 0.0001) { failures++; print("FAIL: cpy data"); }

	// --- mat_map ---
	matrix::mat_map(a) { x => x * 2.0 };
	if !approx(a.data[0], 2.0, 0.0001) { failures++; print("FAIL: mat_map data"); }

	// --- weights: w_init with seed ---
	let w = weights::w_init(2, 3, 7);
	if w.rows != 2 || w.cols != 3 { failures++; print("FAIL: w_init dims"); }
	let w2 = weights::w_init(2, 3, 7);
	var wsame = true;
	for t in 6 { if w.data[t] != w2.data[t] { wsame = false; } }
	if !wsame { failures++; print("FAIL: w_init not reproducible"); }

	// --- weights: w_set ---
	let wok = weights::w_set(w, 0, 0, [99.0]);
	if !wok { failures++; print("FAIL: w_set returned false"); }
	if !approx(w.data[0], 99.0, 0.0001) { failures++; print("FAIL: w_set data"); }
	let wok2 = weights::w_set(w, 5, 0, [1.0]);
	if wok2 { failures++; print("FAIL: w_set bad row should be false"); }

	// --- weights: w_apply ---
	var winback: f64[>..] = [];
	winback.push(1.0); winback.push(2.0);
	winback.push(3.0); winback.push(4.0);
	winback.push(5.0); winback.push(6.0);
	var win_row0: f64[:] = winback[0..2];
	var win_row1: f64[:] = winback[2..4];
	var win_row2: f64[:] = winback[4..6];
	var win_data: f64[:][>..] = []; win_data.push(win_row0); win_data.push(win_row1); win_data.push(win_row2);
	let win, winok = matrix::mat(3, 2, copy(win_data));
	if !winok { failures++; print("FAIL: mat win returned false"); }
	let wc, wcok = weights::w_apply(w, win);
	if !wcok { failures++; print("FAIL: w_apply returned false"); }

	// --- weights: w_apply dimension mismatch ---
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
	if !winpok { failures++; print("FAIL: mat wbad_input returned false"); }
	let wc2, wc2ok = weights::w_apply(wbad, wbad_input);
	if wc2ok { failures++; print("FAIL: w_apply mismatch should be false"); }

	// --- bias: b_init ---
	let bb = bias::b_init(2);
	if bb.rows != 1 || bb.cols != 2 { failures++; print("FAIL: b_init dims"); }
	for t in 2 { if bb.data[t] != 0.0 { failures++; print("FAIL: b_init not zero"); break; } }

	// --- bias: b_set ---
	let bok = bias::b_set(bb, 0, [1.0, 2.0]);
	if !bok { failures++; print("FAIL: b_set returned false"); }
	if !approx(bb.data[0], 1.0, 0.0001) { failures++; print("FAIL: b_set data[0]"); }
	if !approx(bb.data[1], 2.0, 0.0001) { failures++; print("FAIL: b_set data[1]"); }
	let bok2 = bias::b_set(bb, 5, [1.0]);
	if bok2 { failures++; print("FAIL: b_set bad col should be false"); }

	// --- bias: b_add ---
	let ba, baok = bias::b_add(a, bb);
	if !baok { failures++; print("FAIL: b_add returned false"); }
	if !approx(ba.data[0], 3.0, 0.0001) { failures++; print("FAIL: b_add[0]"); }

	// --- bias: b_add dimension mismatch ---
	let bbad = bias::b_init(5);
	let ba2, ba2ok = bias::b_add(a, bbad);
	if ba2ok { failures++; print("FAIL: b_add mismatch should be false"); }

	// --- dense forward (two layers) ---
	var w1back: f64[>..] = [];
	w1back.push(1.0); w1back.push(2.0);
	var w1_row0: f64[:] = w1back[0..2];
	var w1_data: f64[:][>..] = []; w1_data.push(w1_row0);
	let w1, w1ok = matrix::mat(1, 2, copy(w1_data));
	if !w1ok { failures++; print("FAIL: mat w1 returned false"); }
	var b1back: f64[>..] = [];
	b1back.push(3.0);
	var b1_row0: f64[:] = b1back[0..1];
	var b1_data: f64[:][>..] = []; b1_data.push(b1_row0);
	let b1, b1ok = matrix::mat(1, 1, copy(b1_data));
	if !b1ok { failures++; print("FAIL: mat b1 returned false"); }
	var w2back: f64[>..] = [];
	w2back.push(2.0);
	var w2_row0: f64[:] = w2back[0..1];
	var w2_data: f64[:][>..] = []; w2_data.push(w2_row0);
	let w2, w2ok = matrix::mat(1, 1, copy(w2_data));
	if !w2ok { failures++; print("FAIL: mat w2 returned false"); }
	var b2back: f64[>..] = [];
	b2back.push(1.0);
	var b2_row0: f64[:] = b2back[0..1];
	var b2_data: f64[:][>..] = []; b2_data.push(b2_row0);
	let b2, b2ok = matrix::mat(1, 1, copy(b2_data));
	if !b2ok { failures++; print("FAIL: mat b2 returned false"); }

	var xin_data: f64[>..] = [];
	xin_data.push(1.0);
	xin_data.push(2.0);
	let xin: matrix::Mat = matrix::Mat { rows: 2, cols: 1, data: copy(xin_data) };

	// Layer 1: z1 = w1·x + b1, then relu
	let z1, z1ok = matrix::dense_forward(w1, xin, b1);
	if !z1ok { failures++; print("FAIL: dense_forward layer 1 returned false"); }
	matrix::mat_map(z1) { x => if x < 0.0 { 0.0 } else { x } };
	// Layer 2: z2 = w2·z1 + b2, then relu
	let z2, z2ok = matrix::dense_forward(w2, z1, b2);
	if !z2ok { failures++; print("FAIL: dense_forward layer 2 returned false"); }
	matrix::mat_map(z2) { x => if x < 0.0 { 0.0 } else { x } };
	if z2.rows != 1 || z2.cols != 1 { failures++; print("FAIL: two-layer forward dims"); }
	// z1 = [1,2]·[[1],[2]] + [3] = [[5]] + [3] = [[8]]
	// z2 = [[2]]·[[8]] + [1] = [[16]] + [1] = [[17]]
	if !approx(z2.data[0], 17.0, 0.0001) { failures++; print("FAIL: two-layer forward[0]"); }

	// --- dense backward ---
	var wback: f64[>..] = [];
	wback.push(1.0); wback.push(2.0); wback.push(3.0);
	wback.push(4.0); wback.push(5.0); wback.push(6.0);
	var w_row0: f64[:] = wback[0..3];
	var w_row1: f64[:] = wback[3..6];
	var w_data: f64[:][>..] = []; w_data.push(w_row0); w_data.push(w_row1);
	let wk, wkok = matrix::mat(2, 3, copy(w_data));
	if !wkok { failures++; print("FAIL: mat wk returned false"); }
	var xback: f64[>..] = [];
	xback.push(7.0); xback.push(8.0);
	xback.push(9.0); xback.push(10.0);
	xback.push(11.0); xback.push(12.0);
	var x_row0: f64[:] = xback[0..2];
	var x_row1: f64[:] = xback[2..4];
	var x_row2: f64[:] = xback[4..6];
	var x_data: f64[:][>..] = []; x_data.push(x_row0); x_data.push(x_row1); x_data.push(x_row2);
	let x, xok = matrix::mat(3, 2, copy(x_data));
	if !xok { failures++; print("FAIL: mat x returned false"); }
	var dzback: f64[>..] = [];
	dzback.push(1.0); dzback.push(2.0);
	dzback.push(3.0); dzback.push(4.0);
	var dz_row0: f64[:] = dzback[0..2];
	var dz_row1: f64[:] = dzback[2..4];
	var dz_data: f64[:][>..] = []; dz_data.push(dz_row0); dz_data.push(dz_row1);
	let dz, dzok = matrix::mat(2, 2, copy(dz_data));
	if !dzok { failures++; print("FAIL: mat dz returned false"); }
	let dw, db, dx, dok = matrix::dense_backward(wk, x, dz);
	if !dok { failures++; print("FAIL: dense_backward returned false"); }
	if dw.rows != 2 || dw.cols != 3 { failures++; print("FAIL: dense_backward dW dims"); }
	if !approx(dw.data[0], 23.0, 0.0001) { failures++; print("FAIL: dense_backward dW[0]"); }
	if !approx(dw.data[1], 29.0, 0.0001) { failures++; print("FAIL: dense_backward dW[1]"); }
	if !approx(dw.data[2], 35.0, 0.0001) { failures++; print("FAIL: dense_backward dW[2]"); }
	if !approx(dw.data[3], 53.0, 0.0001) { failures++; print("FAIL: dense_backward dW[3]"); }
	if !approx(dw.data[4], 67.0, 0.0001) { failures++; print("FAIL: dense_backward dW[4]"); }
	if !approx(dw.data[5], 81.0, 0.0001) { failures++; print("FAIL: dense_backward dW[5]"); }
	if db.len != 2 { failures++; print("FAIL: dense_backward db len"); }
	if !approx(db[0], 4.0, 0.0001) { failures++; print("FAIL: dense_backward db[0]"); }
	if !approx(db[1], 6.0, 0.0001) { failures++; print("FAIL: dense_backward db[1]"); }
	if dx.rows != 3 || dx.cols != 2 { failures++; print("FAIL: dense_backward dx dims"); }
	if !approx(dx.data[0], 13.0, 0.0001) { failures++; print("FAIL: dense_backward dx[0]"); }
	if !approx(dx.data[1], 18.0, 0.0001) { failures++; print("FAIL: dense_backward dx[1]"); }
	if !approx(dx.data[2], 17.0, 0.0001) { failures++; print("FAIL: dense_backward dx[2]"); }
	if !approx(dx.data[3], 24.0, 0.0001) { failures++; print("FAIL: dense_backward dx[3]"); }
	if !approx(dx.data[4], 21.0, 0.0001) { failures++; print("FAIL: dense_backward dx[4]"); }
	if !approx(dx.data[5], 30.0, 0.0001) { failures++; print("FAIL: dense_backward dx[5]"); }

	// --- functions: mat_relu ---
	var rrback: f64[>..] = [];
	rrback.push(-1.0); rrback.push(2.0);
	rrback.push(-3.0); rrback.push(4.0);
	var rr_row0: f64[:] = rrback[0..2];
	var rr_row1: f64[:] = rrback[2..4];
	var rrdata: f64[:][>..] = []; rrdata.push(rr_row0); rrdata.push(rr_row1);
	let rm, rmok = matrix::mat(2, 2, copy(rrdata));
	if !rmok { failures++; print("FAIL: mat rm returned false"); }
	functions::mat_relu(rm);
	if !approx(rm.data[0], 0.0, 0.0001) { failures++; print("FAIL: relu[0]"); }
	if !approx(rm.data[1], 2.0, 0.0001) { failures++; print("FAIL: relu[1]"); }
	if !approx(rm.data[2], 0.0, 0.0001) { failures++; print("FAIL: relu[2]"); }
	if !approx(rm.data[3], 4.0, 0.0001) { failures++; print("FAIL: relu[3]"); }

	// --- functions: mat_sigmoid ---
	var srback: f64[>..] = [];
	srback.push(0.0);
	var sr_row0: f64[:] = srback[0..1];
	var srdata: f64[:][>..] = []; srdata.push(sr_row0);
	let sm, smok = matrix::mat(1, 1, copy(srdata));
	if !smok { failures++; print("FAIL: mat sm returned false"); }
	functions::mat_sigmoid(sm);
	if !approx(sm.data[0], 0.5, 0.0001) { failures++; print("FAIL: sigmoid[0]"); }

	// --- functions: mat_tanh ---
	var trback: f64[>..] = [];
	trback.push(0.0);
	var tr_row0: f64[:] = trback[0..1];
	var trdata: f64[:][>..] = []; trdata.push(tr_row0);
	let tm, ttok = matrix::mat(1, 1, copy(trdata));
	if !ttok { failures++; print("FAIL: mat tm returned false"); }
	functions::mat_tanh(tm);
	if !approx(tm.data[0], 0.0, 0.0001) { failures++; print("FAIL: tanh[0]"); }

	// --- functions: mat_softmax_rows ---
	var smback: f64[>..] = [];
	smback.push(0.0); smback.push(0.0);
	var sm_row0: f64[:] = smback[0..2];
	var smdata: f64[:][>..] = []; smdata.push(sm_row0);
	let smm, smmok = matrix::mat(1, 2, copy(smdata));
	if !smmok { failures++; print("FAIL: mat smm returned false"); }
	let sok = functions::mat_softmax_rows(smm);
	if !sok { failures++; print("FAIL: softmax returned false"); }
	let ssum = smm.data[0] + smm.data[1];
	if !approx(ssum, 1.0, 0.0001) { failures++; print("FAIL: softmax sum"); }

	// --- softmax with underflow (all very negative) ---
	var sfback: f64[>..] = [];
	sfback.push(-1000.0); sfback.push(-1000.0);
	var sf_row0: f64[:] = sfback[0..2];
	var sfdata: f64[:][>..] = []; sfdata.push(sf_row0);
	let smf, smfok = matrix::mat(1, 2, copy(sfdata));
	if !smfok { failures++; print("FAIL: mat smf returned false"); }
	let sfok = functions::mat_softmax_rows(smf);
	if sfok { failures++; print("FAIL: softmax underflow should be false"); }

	// --- mat_save / mat_load ---
	var save_data: f64[>..] = [];
	save_data.push(1.0); save_data.push(2.0);
	save_data.push(3.0); save_data.push(4.0);
	var save_row0: f64[:] = save_data[0..2];
	var save_row1: f64[:] = save_data[2..4];
	var save_mdata: f64[:][>..] = []; save_mdata.push(save_row0); save_mdata.push(save_row1);
	let save_mat, smok = matrix::mat(2, 2, copy(save_mdata));
	if !smok { failures++; print("FAIL: mat save_mat returned false"); }
	let save_ok = matrix::mat_save(save_mat, "/tmp/test_matrix.txt");
	if !save_ok { failures++; print("FAIL: mat_save returned false"); }
	let load_mat, lok = matrix::mat_load("/tmp/test_matrix.txt");
	if !lok { failures++; print("FAIL: mat_load returned false"); }
	if load_mat.rows != 2 || load_mat.cols != 2 { failures++; print("FAIL: mat_load dims"); }
	if !approx(load_mat.data[0], 1.0, 0.0001) { failures++; print("FAIL: mat_load data[0]"); }
	if !approx(load_mat.data[1], 2.0, 0.0001) { failures++; print("FAIL: mat_load data[1]"); }
	if !approx(load_mat.data[2], 3.0, 0.0001) { failures++; print("FAIL: mat_load data[2]"); }
	if !approx(load_mat.data[3], 4.0, 0.0001) { failures++; print("FAIL: mat_load data[3]"); }

	// --- dense train step ---
	var twback: f64[>..] = [];
	twback.push(1.0); twback.push(0.0);
	var tw_row0: f64[:] = twback[0..2];
	var tw_data: f64[:][>..] = []; tw_data.push(tw_row0);
	let tw, twok = matrix::mat(1, 2, copy(tw_data));
	if !twok { failures++; print("FAIL: mat tw returned false"); }
	var tbback: f64[>..] = [];
	tbback.push(0.0);
	var tb_row0: f64[:] = tbback[0..1];
	var tb_data: f64[:][>..] = []; tb_data.push(tb_row0);
	let tb, tbok = matrix::mat(1, 1, copy(tb_data));
	if !tbok { failures++; print("FAIL: mat tb returned false"); }
	var txback: f64[>..] = [];
	txback.push(2.0); txback.push(3.0);
	var tx_row0: f64[:] = txback[0..1];
	var tx_row1: f64[:] = txback[1..2];
	var tx_data: f64[:][>..] = []; tx_data.push(tx_row0); tx_data.push(tx_row1);
	let tx, txok = matrix::mat(2, 1, copy(tx_data));
	if !txok { failures++; print("FAIL: mat tx returned false"); }
	var tyback: f64[>..] = [];
	tyback.push(10.0);
	var ty_row0: f64[:] = tyback[0..1];
	var ty_data: f64[:][>..] = []; ty_data.push(ty_row0);
	let ty, tyok = matrix::mat(1, 1, copy(ty_data));
	if !tyok { failures++; print("FAIL: mat ty returned false"); }

	let loss1, lok1 = matrix::dense_train_step(tw, tb, tx, ty, 0.01);
	if !lok1 { failures++; print("FAIL: dense_train_step returned false"); }
	let loss2, lok2 = matrix::dense_train_step(tw, tb, tx, ty, 0.01);
	if !lok2 { failures++; print("FAIL: dense_train_step second call returned false"); }
	if loss2 >= loss1 { failures++; print("FAIL: loss did not decrease"); }

	// --- mat_bce ---
	var predback: f64[>..] = [];
	predback.push(0.9); predback.push(0.1);
	predback.push(0.2); predback.push(0.8);
	var pred_row0: f64[:] = predback[0..2];
	var pred_row1: f64[:] = predback[2..4];
	var pred_d: f64[:][>..] = []; pred_d.push(pred_row0); pred_d.push(pred_row1);
	let pred, pok = matrix::mat(2, 2, copy(pred_d));
	if !pok { failures++; print("FAIL: mat pred returned false"); }
	var tyback: f64[>..] = [];
	tyback.push(1.0); tyback.push(0.0);
	tyback.push(0.0); tyback.push(1.0);
	var ty_row0: f64[:] = tyback[0..2];
	var ty_row1: f64[:] = tyback[2..4];
	var ty_d: f64[:][>..] = []; ty_d.push(ty_row0); ty_d.push(ty_row1);
	let ty2, ty2ok = matrix::mat(2, 2, copy(ty_d));
	if !ty2ok { failures++; print("FAIL: mat ty returned false"); }
	let bce, bok = matrix::mat_bce(pred, ty2);
	if !bok { failures++; print("FAIL: mat_bce returned false"); }
	if !approx(bce, 0.16425, 0.001) { failures++; print("FAIL: mat_bce value"); }

	// --- mat_acc ---
	let acc, aok = matrix::mat_acc(pred, ty2);
	if !aok { failures++; print("FAIL: mat_acc returned false"); }
	if !approx(acc, 1.0, 0.0001) { failures++; print("FAIL: mat_acc value"); }

	// --- mat_xavier_init ---
	let wx = matrix::mat_xavier_init(3, 4);
	if wx.rows != 3 || wx.cols != 4 { failures++; print("FAIL: mat_xavier_init dims"); }
	var wx_max = 0.0;
	var wx_min = 1.0;
	for t in 12 {
		if wx.data[t] > wx_max { wx_max = wx.data[t]; }
		if wx.data[t] < wx_min { wx_min = wx.data[t]; }
	}
	if wx_max > 0.926 || wx_min < -0.926 { failures++; print("FAIL: mat_xavier_init range"); }

	// --- mat_he_init ---
	let wh = matrix::mat_he_init(3, 4);
	if wh.rows != 3 || wh.cols != 4 { failures++; print("FAIL: mat_he_init dims"); }
	var wh_max = 0.0;
	var wh_min = 1.0;
	for t in 12 {
		if wh.data[t] > wh_max { wh_max = wh.data[t]; }
		if wh.data[t] < wh_min { wh_min = wh.data[t]; }
	}
	if wh_max > 1.415 || wh_min < -1.415 { failures++; print("FAIL: mat_he_init range"); }

	if failures == 0 { print("ALL TESTS PASSED" ); } else { print("SOME TESTS FAILED"); }
}
