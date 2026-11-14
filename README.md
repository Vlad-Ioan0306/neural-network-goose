Neural Network Library in Goose
================================

A from-scratch neural network library written in the Goose language, with
matrix operations, dense layers, backpropagation, SGD training, activations,
and weight/bias helpers. Includes a demo program that trains a 2→3→1 MLP to
solve XOR.

## Structure

```
src/
  code/include/
    matrix/matrix.goose     Mat struct, matmul, dense_forward/backward,
                            train_step, sum/row/col/copy/map/gen/print/save/load
    weights/weights.goose   w_init, w_set, w_apply
    bias/bias.goose         b_init, b_set, b_add
    functions/functions.goose ReLU, sigmoid, tanh, softmax, mse, bce, acc
    activations.goose       (aliases / convenience wrappers)
  code/test/
    main.goose              Unit tests for the matrix library
  run/
    main.goose              Demo: XOR (2→3→1 MLP) trained via backprop
```

## Build & Run

Goose binary: `GOOSE_BINARY`

```sh
# Build and run the XOR demo
goose build src/run/main.goose && ./src/run/main

# Build and run the unit tests
goose build src/code/test/main.goose && ./src/code/test/main
```

Compiled binaries are gitignored (`src/code/test/main`, `src/run/main`).

## What the demo does

`src/run/main.goose` trains a small MLP (2 inputs → 3 hidden → 1 output) on
the XOR dataset using backpropagation with SGD (50k epochs, lr=0.5). It prints
predictions before and after training, then an expected-vs-output benchmark:

```
=== EXPECTED vs OUTPUT ===

XOR(0,0)
expected: 0
output:   0.0045

XOR(0,1)
expected: 1
output:   0.9955

...

=== BENCHMARK ===
ALL TESTS PASSED
```

## Matrix layout

Goose does not support nested grow-only arrays (`f64[:][:]`), so all matrices
are flat `f64[:]` stored row-major: element `(r, c)` is at index `r * cols + c`.
`mat(rows, cols, data)` takes the flat array directly.

## Language constraints

- Generic functions use a trailing block: `mat_map(m) { x => ... }`
- Grow-only arrays passed by value require `copy()`
- Structs with grow-only fields must be `root struct`
- All statements end with `;`
- No string concatenation with numeric types — print them separately

## License

None specified. Use at your own risk.
