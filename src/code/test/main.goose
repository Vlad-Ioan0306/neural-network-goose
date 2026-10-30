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

fn check(name: const u8[:], ok: bool) -> i64 {
	if ok { print(name, " - ✓ passed"); return 0; }
	print(name, " - ✗ failed"); return 1;
}

fn main() {
	var failures: i64 = 0;

	// --- mat constructor (zero-filled from empty data) ---
	var ok: bool = true;
	var empty: f64[:][>..] = [];
	let m, mok = matrix::mat(2, 3, copy(empty));
	if !mok { ok = false; }
	if m.rows != 2 || m.cols != 3 { ok = false; }
	for t in 2 * 3 {
		if m.data[t] != 0.0 { ok = false; break; }
	}
	failures += check("mat constructor", ok);

	// --- mat from data ---
	ok = true;
	var backing: f64[>..] = [];
	backing.push(1.0); backing.push(2.0); backing.push(3.0);
	backing.push(4.0); backing.push(5.0); backing.push(6.0);
	var row0: f64[:] = backing[0..3];
	var row1: f64[:] = backing[3..6];
	var d2: f64[:][>..] = []; d2.push(row0); d2.push(row1);
	let m2, m2ok = matrix::mat(2, 3, copy(d2));
	if !m2ok { ok = false; }
	if !approx(m2.data[0], 1.0, 0.0001) { ok = false; }
	if !approx(m2.data[5], 6.0, 0.0001) { ok = false; }
	failures += check("mat from data", ok);

	// --- mat from data wrong rows ---
	ok = true;
	var wrong_backing: f64[>..] = [];
	wrong_backing.push(1.0); wrong_backing.push(2.0); wrong_backing.push(3.0);
	var wrong_row: f64[:] = wrong_backing[0..3];
	var wrong_rows: f64[:][>..] = []; wrong_rows.push(wrong_row);
	let m3, m3ok = matrix::mat(2, 3, copy(wrong_rows));
	if m3ok { ok = false; }
	failures += check("mat wrong rows should fail", ok);

	// --- mat from data wrong cols ---
	ok = true;
	var wc_backing: f64[>..] = [];
	wc_backing.push(1.0); wc_backing.push(2.0);
	wc_backing.push(3.0); wc_backing.push(4.0); wc_backing.push(5.0);
	var wc_row1: f64[:] = wc_backing[0..2];
	var wc_row2: f64[:] = wc_backing[2..5];
	var wc_data: f64[:][>..] = []; wc_data.push(wc_row1); wc_data.push(wc_row2);
	let m4, m4ok = matrix::mat(2, 2, copy(wc_data));
	if m4ok { ok = false; }
	failures += check("mat wrong cols should fail", ok);

	// --- mat_gen reproducibility ---
	ok = true;
	let g1 = matrix::mat_gen(2, 2, 42);
	let g2 = matrix::mat_gen(2, 2, 42);
	let g3 = matrix::mat_gen(2, 2, 99);
	var same = true;
	for t in 4 { if g1.data[t] != g2.data[t] { same = false; } }
	if !same { ok = false; }
	var diff = false;
	for t in 4 { if g1.data[t] != g3.data[t] { diff = true; } }
	if !diff { ok = false; }
	failures += check("mat_gen reproducibility", ok);

	// --- matmul with known values ---
	ok = true;
	var aback: f64[>..] = [];
	aback.push(1.0); aback.push(2.0);
	aback.push(3.0); aback.push(4.0);
	var a_row0: f64[:] = aback[0..2];
	var a_row1: f64[:] = aback[2..4];
	var adata: f64[:][>..] = []; adata.push(a_row0); adata.push(a_row1);
	let a, aok = matrix::mat(2, 2, copy(adata));
	if !aok { ok = false; }
	var bback: f64[>..] = [];
	bback.push(5.0); bback.push(6.0);
	bback.push(7.0); bback.push(8.0);
	var b_row0: f64[:] = bback[0..2];
	var b_row1: f64[:] = bback[2..4];
	var bdata: f64[:][>..] = []; bdata.push(b_row0); bdata.push(b_row1);
	let b, bok = matrix::mat(2, 2, copy(bdata));
	if !bok { ok = false; }
	let c, cok = matrix::matmul(a, b);
	if !cok { ok = false; }
	if c.rows != 2 || c.cols != 2 { ok = false; }
	if !approx(c.data[0], 19.0, 0.0001) { ok = false; }
	if !approx(c.data[1], 22.0, 0.0001) { ok = false; }
	if !approx(c.data[2], 43.0, 0.0001) { ok = false; }
	if !approx(c.data[3], 50.0, 0.0001) { ok = false; }
	failures += check("matmul known values", ok);

	// --- matmul dimension mismatch ---
	ok = true;
	var badback: f64[>..] = [];
	badback.push(1.0); badback.push(2.0);
	badback.push(3.0); badback.push(4.0);
	badback.push(5.0); badback.push(6.0);
	var bad_row0: f64[:] = badback[0..2];
	var bad_row1: f64[:] = badback[2..4];
	var bad_row2: f64[:] = badback[4..6];
	var bad_data: f64[:][>..] = []; bad_data.push(bad_row0); bad_data.push(bad_row1); bad_data.push(bad_row2);
	let bad, badok = matrix::mat(3, 2, copy(bad_data));
	if !badok { ok = false; }
	let c2, c2ok = matrix::matmul(a, bad);
	if c2ok { ok = false; }
	if c2.rows != 0 { ok = false; }
	failures += check("matmul dimension mismatch", ok);

	// --- mat_add ---
	ok = true;
	let d_add, dok = matrix::mat_add(a, b);
	if !dok { ok = false; }
	if !approx(d_add.data[0], 6.0, 0.0001) { ok = false; }
	failures += check("mat_add", ok);

	// --- mat_mul (Hadamard) ---
	ok = true;
	let e, eok = matrix::mat_mul(a, b);
	if !eok { ok = false; }
	if !approx(e.data[0], 5.0, 0.0001) { ok = false; }
	failures += check("mat_mul", ok);

	// --- mat_div ---
	ok = true;
	let f, fok = matrix::mat_div(a, b);
	if !fok { ok = false; }
	if !approx(f.data[0], 0.2, 0.0001) { ok = false; }
	failures += check("mat_div", ok);

	// --- mat_div by zero ---
	ok = true;
	var bzback: f64[>..] = [];
	bzback.push(0.0); bzback.push(1.0);
	bzback.push(1.0); bzback.push(1.0);
	var bz_row0: f64[:] = bzback[0..2];
	var bz_row1: f64[:] = bzback[2..4];
	var bzdata: f64[:][>..] = []; bzdata.push(bz_row0); bzdata.push(bz_row1);
	let bz, bzok = matrix::mat(2, 2, copy(bzdata));
	if !bzok { ok = false; }
	let g, gok = matrix::mat_div(a, bz);
	if gok { ok = false; }
	failures += check("mat_div by zero should fail", ok);

	// --- mat_sum_rows ---
	ok = true;
	let sums = matrix::mat_sum_rows(a);
	if sums[0] != 3.0 { ok = false; }
	if sums[1] != 7.0 { ok = false; }
	failures += check("mat_sum_rows", ok);

	// --- mat_sum_cols ---
	ok = true;
	let sumc = matrix::mat_sum_cols(a);
	if sumc[0] != 4.0 { ok = false; }
	if sumc[1] != 6.0 { ok = false; }
	failures += check("mat_sum_cols", ok);

	// --- mat_row ---
	ok = true;
	let r, rok = matrix::mat_row(a, 0);
	if !rok { ok = false; }
	if r[0] != 1.0 || r[1] != 2.0 { ok = false; }
	let r2, r2ok = matrix::mat_row(a, 5);
	if r2ok { ok = false; }
	failures += check("mat_row", ok);

	// --- mat_col ---
	ok = true;
	let ccol, ccolok = matrix::mat_col(a, 0);
	if !ccolok { ok = false; }
	if ccol[0] != 1.0 || ccol[1] != 3.0 { ok = false; }
	failures += check("mat_col", ok);

	// --- mat_reshape ---
	ok = true;
	let reshaped, rokok = matrix::mat_reshape(a, 1, 4);
	if !rokok { ok = false; }
	if reshaped.rows != 1 || reshaped.cols != 4 { ok = false; }
	let badreshape, brkok = matrix::mat_reshape(a, 3, 3);
	if brkok { ok = false; }
	failures += check("mat_reshape", ok);

	// --- mat_concat_rows ---
	ok = true;
	let cr, crok = matrix::mat_concat_rows(a, b);
	if !crok { ok = false; }
	if cr.rows != 4 || cr.cols != 2 { ok = false; }
	failures += check("mat_concat_rows", ok);

	// --- mat_concat_cols ---
	ok = true;
	let cc, cck = matrix::mat_concat_cols(a, b);
	if !cck { ok = false; }
	if cc.rows != 2 || cc.cols != 4 { ok = false; }
	failures += check("mat_concat_cols", ok);

	// --- transpose ---
	ok = true;
	let bt = matrix::transpose(a);
	if bt.rows != 2 || bt.cols != 2 { ok = false; }
	if !approx(bt.data[0], 1.0, 0.0001) { ok = false; }
	if !approx(bt.data[1], 3.0, 0.0001) { ok = false; }
	failures += check("transpose", ok);

	// --- mat_cpy ---
	ok = true;
	let cp = matrix::mat_cpy(a);
	if cp.rows != 2 || cp.cols != 2 { ok = false; }
	if !approx(cp.data[0], 1.0, 0.0001) { ok = false; }
	failures += check("mat_cpy", ok);

	// --- mat_map ---
	ok = true;
	matrix::mat_map(a) { x => x * 2.0 };
	if !approx(a.data[0], 2.0, 0.0001) { ok = false; }
	failures += check("mat_map", ok);

	// --- weights: w_init with seed ---
	ok = true;
	let w = weights::w_init(2, 3, 7);
	if w.rows != 2 || w.cols != 3 { ok = false; }
	let w2 = weights::w_init(2, 3, 7);
	var wsame = true;
	for t in 6 { if w.data[t] != w2.data[t] { wsame = false; } }
	if !wsame { ok = false; }
	failures += check("w_init reproducibility", ok);

	// --- weights: w_set ---
	ok = true;
	let wok = weights::w_set(w, 0, 0, [99.0]);
	if !wok { ok = false; }
	if !approx(w.data[0], 99.0, 0.0001) { ok = false; }
	let wok2 = weights::w_set(w, 5, 0, [1.0]);
	if wok2 { ok = false; }
	failures += check("w_set", ok);

	// --- weights: w_apply ---
	ok = true;
	var winback: f64[>..] = [];
	winback.push(1.0); winback.push(2.0);
	winback.push(3.0); winback.push(4.0);
	winback.push(5.0); winback.push(6.0);
	var win_row0: f64[:] = winback[0..2];
	var win_row1: f64[:] = winback[2..4];
	var win_row2: f64[:] = winback[4..6];
	var win_data: f64[:][>..] = []; win_data.push(win_row0); win_data.push(win_row1); win_data.push(win_row2);
	let win, winok = matrix::mat(3, 2, copy(win_data));
	if !winok { ok = false; }
	let wc, wcok = weights::w_apply(w, win);
	if !wcok { ok = false; }
	failures += check("w_apply", ok);

	// --- weights: w_apply dimension mismatch ---
	ok = true;
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
	if !winpok { ok = false; }
	let wc2, wc2ok = weights::w_apply(wbad, wbad_input);
	if wc2ok { ok = false; }
	failures += check("w_apply dimension mismatch should fail", ok);

	// --- bias: b_init ---
	ok = true;
	let bb = bias::b_init(2);
	if bb.rows != 1 || bb.cols != 2 { ok = false; }
	for t in 2 { if bb.data[t] != 0.0 { ok = false; break; } }
	failures += check("b_init", ok);

	// --- bias: b_set ---
	ok = true;
	let bok = bias::b_set(bb, 0, [1.0, 2.0]);
	if !bok { ok = false; }
	if !approx(bb.data[0], 1.0, 0.0001) { ok = false; }
	if !approx(bb.data[1], 2.0, 0.0001) { ok = false; }
	let bok2 = bias::b_set(bb, 5, [1.0]);
	if bok2 { ok = false; }
	failures += check("b_set", ok);

	// --- bias: b_add ---
	ok = true;
	let ba, baok = bias::b_add(a, bb);
	if !baok { ok = false; }
	if !approx(ba.data[0], 3.0, 0.0001) { ok = false; }
	failures += check("b_add", ok);

	// --- bias: b_add dimension mismatch ---
	ok = true;
	let bbad = bias::b_init(5);
	let ba2, ba2ok = bias::b_add(a, bbad);
	if ba2ok { ok = false; }
	failures += check("b_add dimension mismatch should fail", ok);

	// --- dense forward (two layers) ---
	ok = true;
	var w1back: f64[>..] = [];
	w1back.push(1.0); w1back.push(2.0);
	var w1_row0: f64[:] = w1back[0..2];
	var w1_data: f64[:][>..] = []; w1_data.push(w1_row0);
	let w1, w1ok = matrix::mat(1, 2, copy(w1_data));
	if !w1ok { ok = false; }
	var b1back: f64[>..] = [];
	b1back.push(3.0);
	var b1_row0: f64[:] = b1back[0..1];
	var b1_data: f64[:][>..] = []; b1_data.push(b1_row0);
	let b1, b1ok = matrix::mat(1, 1, copy(b1_data));
	if !b1ok { ok = false; }
	var w2back: f64[>..] = [];
	w2back.push(2.0);
	var w2_row0: f64[:] = w2back[0..1];
	var w2_data: f64[:][>..] = []; w2_data.push(w2_row0);
	let w2, w2ok = matrix::mat(1, 1, copy(w2_data));
	if !w2ok { ok = false; }
	var b2back: f64[>..] = [];
	b2back.push(1.0);
	var b2_row0: f64[:] = b2back[0..1];
	var b2_data: f64[:][>..] = []; b2_data.push(b2_row0);
	let b2, b2ok = matrix::mat(1, 1, copy(b2_data));
	if !b2ok { ok = false; }

	var xin_data: f64[>..] = [];
	xin_data.push(1.0);
	xin_data.push(2.0);
	let xin: matrix::Mat = matrix::Mat { rows: 2, cols: 1, data: copy(xin_data) };

	let z1, z1ok = matrix::dense_forward(w1, xin, b1);
	if !z1ok { ok = false; }
	matrix::mat_map(z1) { x => if x < 0.0 { 0.0 } else { x } };
	let z2, z2ok = matrix::dense_forward(w2, z1, b2);
	if !z2ok { ok = false; }
	matrix::mat_map(z2) { x => if x < 0.0 { 0.0 } else { x } };
	if z2.rows != 1 || z2.cols != 1 { ok = false; }
	if !approx(z2.data[0], 17.0, 0.0001) { ok = false; }
	failures += check("dense forward (two layers)", ok);

	// --- dense backward ---
	ok = true;
	var wback: f64[>..] = [];
	wback.push(1.0); wback.push(2.0); wback.push(3.0);
	wback.push(4.0); wback.push(5.0); wback.push(6.0);
	var w_row0: f64[:] = wback[0..3];
	var w_row1: f64[:] = wback[3..6];
	var w_data: f64[:][>..] = []; w_data.push(w_row0); w_data.push(w_row1);
	let wk, wkok = matrix::mat(2, 3, copy(w_data));
	if !wkok { ok = false; }
	var xback: f64[>..] = [];
	xback.push(7.0); xback.push(8.0);
	xback.push(9.0); xback.push(10.0);
	xback.push(11.0); xback.push(12.0);
	var x_row0: f64[:] = xback[0..2];
	var x_row1: f64[:] = xback[2..4];
	var x_row2: f64[:] = xback[4..6];
	var x_data: f64[:][>..] = []; x_data.push(x_row0); x_data.push(x_row1); x_data.push(x_row2);
	let x, xok = matrix::mat(3, 2, copy(x_data));
	if !xok { ok = false; }
	var dzback: f64[>..] = [];
	dzback.push(1.0); dzback.push(2.0);
	dzback.push(3.0); dzback.push(4.0);
	var dz_row0: f64[:] = dzback[0..2];
	var dz_row1: f64[:] = dzback[2..4];
	var dz_data: f64[:][>..] = []; dz_data.push(dz_row0); dz_data.push(dz_row1);
	let dz, dzok = matrix::mat(2, 2, copy(dz_data));
	if !dzok { ok = false; }
	let dw, db, dx, dok = matrix::dense_backward(wk, x, dz);
	if !dok { ok = false; }
	if dw.rows != 2 || dw.cols != 3 { ok = false; }
	if !approx(dw.data[0], 23.0, 0.0001) { ok = false; }
	if !approx(dw.data[1], 29.0, 0.0001) { ok = false; }
	if !approx(dw.data[2], 35.0, 0.0001) { ok = false; }
	if !approx(dw.data[3], 53.0, 0.0001) { ok = false; }
	if !approx(dw.data[4], 67.0, 0.0001) { ok = false; }
	if !approx(dw.data[5], 81.0, 0.0001) { ok = false; }
	if db.len != 2 { ok = false; }
	if !approx(db[0], 4.0, 0.0001) { ok = false; }
	if !approx(db[1], 6.0, 0.0001) { ok = false; }
	if dx.rows != 3 || dx.cols != 2 { ok = false; }
	if !approx(dx.data[0], 13.0, 0.0001) { ok = false; }
	if !approx(dx.data[1], 18.0, 0.0001) { ok = false; }
	if !approx(dx.data[2], 17.0, 0.0001) { ok = false; }
	if !approx(dx.data[3], 24.0, 0.0001) { ok = false; }
	if !approx(dx.data[4], 21.0, 0.0001) { ok = false; }
	if !approx(dx.data[5], 30.0, 0.0001) { ok = false; }
	failures += check("dense backward", ok);

	// --- functions: mat_relu ---
	ok = true;
	var rrback: f64[>..] = [];
	rrback.push(-1.0); rrback.push(2.0);
	rrback.push(-3.0); rrback.push(4.0);
	var rr_row0: f64[:] = rrback[0..2];
	var rr_row1: f64[:] = rrback[2..4];
	var rrdata: f64[:][>..] = []; rrdata.push(rr_row0); rrdata.push(rr_row1);
	let rm, rmok = matrix::mat(2, 2, copy(rrdata));
	if !rmok { ok = false; }
	functions::mat_relu(rm);
	if !approx(rm.data[0], 0.0, 0.0001) { ok = false; }
	if !approx(rm.data[1], 2.0, 0.0001) { ok = false; }
	if !approx(rm.data[2], 0.0, 0.0001) { ok = false; }
	if !approx(rm.data[3], 4.0, 0.0001) { ok = false; }
	failures += check("mat_relu", ok);

	// --- functions: mat_sigmoid ---
	ok = true;
	var srback: f64[>..] = [];
	srback.push(0.0);
	var sr_row0: f64[:] = srback[0..1];
	var srdata: f64[:][>..] = []; srdata.push(sr_row0);
	let sm, smok = matrix::mat(1, 1, copy(srdata));
	if !smok { ok = false; }
	functions::mat_sigmoid(sm);
	if !approx(sm.data[0], 0.5, 0.0001) { ok = false; }
	failures += check("mat_sigmoid", ok);

	// --- functions: mat_tanh ---
	ok = true;
	var trback: f64[>..] = [];
	trback.push(0.0);
	var tr_row0: f64[:] = trback[0..1];
	var trdata: f64[:][>..] = []; trdata.push(tr_row0);
	let tm, ttok = matrix::mat(1, 1, copy(trdata));
	if !ttok { ok = false; }
	functions::mat_tanh(tm);
	if !approx(tm.data[0], 0.0, 0.0001) { ok = false; }
	failures += check("mat_tanh", ok);

	// --- functions: mat_softmax_rows ---
	ok = true;
	var smback: f64[>..] = [];
	smback.push(0.0); smback.push(0.0);
	var sm_row0: f64[:] = smback[0..2];
	var smdata: f64[:][>..] = []; smdata.push(sm_row0);
	let smm, smmok = matrix::mat(1, 2, copy(smdata));
	if !smmok { ok = false; }
	let sok = functions::mat_softmax_rows(smm);
	if !sok { ok = false; }
	let ssum = smm.data[0] + smm.data[1];
	if !approx(ssum, 1.0, 0.0001) { ok = false; }
	failures += check("mat_softmax_rows", ok);

	// --- softmax with underflow (all very negative) ---
	ok = true;
	var sfback: f64[>..] = [];
	sfback.push(-1000.0); sfback.push(-1000.0);
	var sf_row0: f64[:] = sfback[0..2];
	var sfdata: f64[:][>..] = []; sfdata.push(sf_row0);
	let smf, smfok = matrix::mat(1, 2, copy(sfdata));
	if !smfok { ok = false; }
	let sfok = functions::mat_softmax_rows(smf);
	if sfok { ok = false; }
	failures += check("softmax underflow should fail", ok);

	// --- mat_save / mat_load ---
	ok = true;
	var save_data: f64[>..] = [];
	save_data.push(1.0); save_data.push(2.0);
	save_data.push(3.0); save_data.push(4.0);
	var save_row0: f64[:] = save_data[0..2];
	var save_row1: f64[:] = save_data[2..4];
	var save_mdata: f64[:][>..] = []; save_mdata.push(save_row0); save_mdata.push(save_row1);
	let save_mat, smok = matrix::mat(2, 2, copy(save_mdata));
	if !smok { ok = false; }
	let save_ok = matrix::mat_save(save_mat, "/tmp/test_matrix.txt");
	if !save_ok { ok = false; }
	let load_mat, lok = matrix::mat_load("/tmp/test_matrix.txt");
	if !lok { ok = false; }
	if load_mat.rows != 2 || load_mat.cols != 2 { ok = false; }
	if !approx(load_mat.data[0], 1.0, 0.0001) { ok = false; }
	if !approx(load_mat.data[1], 2.0, 0.0001) { ok = false; }
	if !approx(load_mat.data[2], 3.0, 0.0001) { ok = false; }
	if !approx(load_mat.data[3], 4.0, 0.0001) { ok = false; }
	failures += check("mat_save / mat_load", ok);

	// --- dense train step ---
	ok = true;
	var twback: f64[>..] = [];
	twback.push(1.0); twback.push(0.0);
	var tw_row0: f64[:] = twback[0..2];
	var tw_data: f64[:][>..] = []; tw_data.push(tw_row0);
	let tw, twok = matrix::mat(1, 2, copy(tw_data));
	if !twok { ok = false; }
	var tbback: f64[>..] = [];
	tbback.push(0.0);
	var tb_row0: f64[:] = tbback[0..1];
	var tb_data: f64[:][>..] = []; tb_data.push(tb_row0);
	let tb, tbok = matrix::mat(1, 1, copy(tb_data));
	if !tbok { ok = false; }
	var txback: f64[>..] = [];
	txback.push(2.0); txback.push(3.0);
	var tx_row0: f64[:] = txback[0..1];
	var tx_row1: f64[:] = txback[1..2];
	var tx_data: f64[:][>..] = []; tx_data.push(tx_row0); tx_data.push(tx_row1);
	let tx, txok = matrix::mat(2, 1, copy(tx_data));
	if !txok { ok = false; }
	var tyback: f64[>..] = [];
	tyback.push(10.0);
	var ty_row0: f64[:] = tyback[0..1];
	var ty_data: f64[:][>..] = []; ty_data.push(ty_row0);
	let ty, tyok = matrix::mat(1, 1, copy(ty_data));
	if !tyok { ok = false; }

	let loss1, lok1 = matrix::dense_train_step(tw, tb, tx, ty, 0.01);
	if !lok1 { ok = false; }
	let loss2, lok2 = matrix::dense_train_step(tw, tb, tx, ty, 0.01);
	if !lok2 { ok = false; }
	if loss2 >= loss1 { ok = false; }
	failures += check("dense_train_step", ok);

	// --- mat_bce ---
	ok = true;
	var predback: f64[>..] = [];
	predback.push(0.9); predback.push(0.1);
	predback.push(0.2); predback.push(0.8);
	var pred_row0: f64[:] = predback[0..2];
	var pred_row1: f64[:] = predback[2..4];
	var pred_d: f64[:][>..] = []; pred_d.push(pred_row0); pred_d.push(pred_row1);
	let pred, pok = matrix::mat(2, 2, copy(pred_d));
	if !pok { ok = false; }
	var tyback: f64[>..] = [];
	tyback.push(1.0); tyback.push(0.0);
	tyback.push(0.0); tyback.push(1.0);
	var ty_row0: f64[:] = tyback[0..2];
	var ty_row1: f64[:] = tyback[2..4];
	var ty_d: f64[:][>..] = []; ty_d.push(ty_row0); ty_d.push(ty_row1);
	let ty2, ty2ok = matrix::mat(2, 2, copy(ty_d));
	if !ty2ok { ok = false; }
	let bce, bok = matrix::mat_bce(pred, ty2);
	if !bok { ok = false; }
	if !approx(bce, 0.16425, 0.001) { ok = false; }
	failures += check("mat_bce", ok);

	// --- mat_acc ---
	ok = true;
	let acc, aok = matrix::mat_acc(pred, ty2);
	if !aok { ok = false; }
	if !approx(acc, 1.0, 0.0001) { ok = false; }
	failures += check("mat_acc", ok);

	// --- mat_xavier_init ---
	ok = true;
	let wx = matrix::mat_xavier_init(3, 4);
	if wx.rows != 3 || wx.cols != 4 { ok = false; }
	var wx_max = 0.0;
	var wx_min = 1.0;
	for t in 12 {
		if wx.data[t] > wx_max { wx_max = wx.data[t]; }
		if wx.data[t] < wx_min { wx_min = wx.data[t]; }
	}
	if wx_max > 0.926 || wx_min < -0.926 { ok = false; }
	failures += check("mat_xavier_init range", ok);

	// --- mat_he_init ---
	ok = true;
	let wh = matrix::mat_he_init(3, 4);
	if wh.rows != 3 || wh.cols != 4 { ok = false; }
	var wh_max = 0.0;
	var wh_min = 1.0;
	for t in 12 {
		if wh.data[t] > wh_max { wh_max = wh.data[t]; }
		if wh.data[t] < wh_min { wh_min = wh.data[t]; }
	}
	if wh_max > 1.415 || wh_min < -1.415 { ok = false; }
	failures += check("mat_he_init range", ok);

	print("");
	if failures == 0 { print("ALL TESTS PASSED"); } else { print("SOME TESTS FAILED (" , failures, ")"); }
}
