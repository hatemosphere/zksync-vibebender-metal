# GKR Compiler and Prover POC

This crate contains a POC GKR circuit compiler and prover.

**To Dos**
- [ ] full Prover
- [ ] full Verifier
- [ ] e2e tests

## Circuit

A `Circuit<F>` (`circuit.rs`) is layered quadratic arithmetic circuit over a field F. Each gate is a quadratic linear combination: `Σ_i coeff_i * wire_b[i] * wire_c[i]`. Layer 0 = outputs, final layer connects to inputs

## Compiler

The compiler (`compiler/`) transforms high-level operations into layered circuits.

e.g. build a circuit for multiplying two inputs
```rust
let mut builder = CircuitBuilder::<F>::new();

let x = builder.input();
let y = builder.input();
let z = builder.mul(x, y);
builder.output(z);

let circuit = builder.compile().unwrap();
```

### `CircuitBuilder`
Intermediate representation of the circuit as a list of nodes, with each node containing references to left and right wires (indices in the list).

**`wire_0`:**
The node at index 0 is special:
- Always evaluates to 1
- Used for constant terms: `c * wire_0 * wire_0 = c`
- Used for linear operations: `x * wire_0 = x`
- Used for copy gates: `wire * wire_0 = wire` - essentialy push a node up a layer


**Common subexpression elimination:** If two identical nodes exist, eliminate one. I.e. this results in one gate
```rust
let x = builder.input();
let y = builder.input();

let xy1 = builder.mul(x, y);
let xy2 = builder.mul(x, y);
let xy3 = builder.mul(x, y);
```

**Constant folding:** Constant operations are computed at compile time. I.e. this results in one constant gate with value 15
```rust
let c1 = builder.constant(3);
let c2 = builder.constant(5);
let product = builder.mul(c1, c2);
builder.output(product);
```

### `Scheduler`
Transforms `CircuitBuilder` into a layered `Circuit`. It determines which layer each node belongs to, remove unused nodes, insert copy wires if a node is needed in multiple layers, and ensures that all wire references are valid and that each layer only depends on the previous layer.

## Sumcheck

Two sumcheck variants (`sumcheck/`):

**Multilinear Sumcheck** (`multilinear_sumcheck.rs`):
- Proves `Σ_x g(x) = c` for multilinear `g`
- Mostly for testing/reference

**Batched Sumcheck** (`batched_sumcheck.rs`):
- Proves multiple sumcheck instances with random linear combination: `Σ_i α_i·c_i = Σ_x Σ_i α_i·g_i(x)`
- Batching coefficients α_1, α_2, ..., α_n drawn from transcript
- **Padding**: Shorter instances emit constant polynomials until their rounds start
- Returns `SumcheckInstanceProof<F, E>` (compressed polys) and final challenges

## Prover

GKR prover (`prover/`):

**LayerSumcheckProver** (`layer_sumcheck.rs`):
- Proves `V_i(z) = Σ_{b,c∈{0,1}^n} add_i(z,b,c)·V_{i+1}(b)·V_{i+1}(c)` for layer i
- Takes: random point z (gate selector), layer definition, previous layer values, claim
- Outputs: sumcheck proof, final challenges `(r_b, r_c)`

**Main Prover Loop** (`mod.rs`):
Not yet implemented