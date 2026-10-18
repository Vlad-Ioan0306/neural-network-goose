// main.goose — runtime tests for matrix, weights, bias, and functions modules.
// Run with: goose build main.goose && ./main

import matrix from "./matrix";
import weights from "./weights";
import bias from "./bias";
import functions from "./functions";

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

	print(failures);
	if failures == 0 { print("ALL TESTS PASSED"); } else { print("SOME TESTS FAILED"); }
}
