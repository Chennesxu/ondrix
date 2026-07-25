# Experimental `.ox` Frontend

`ondrix-compile` is a standalone C++ frontend. Its initial executable surface
accepts one `def` per file. Most current kernels use rank-1 buffers or tensors;
the fixed SOS slice also accepts the explicit rank-2 section layouts described
below. A `def` declares a DSP kernel entry point; it is not a general Python
function.

For a statically bounded Q15 dot, FIR sample, convolution, or correlation,
omitting the accumulator policy requests target-independent exact mathematical
accumulation:

```python
def q15_fir_auto(
    window: buffer[q15, 4], coefficients: buffer[q15, 4]) -> q15:
  return fir(window, coefficients)
```

The frontend derives the smallest signed accumulator width that contains every
possible full-precision Q15 product sum, with a minimum width of 32 bits. Scalar
dot/FIR use the common operand extent; convolution/correlation use the kernel
extent for each output window. The example above normalizes to an i34/frac30
accumulator, while a three-tap correlation uses i33/frac30. This inferred width
is not a hardware register choice, and a target may not silently narrow it.
Dynamic reductions currently require an explicit finite profile because no
finite exact width follows from an unbounded runtime length.

An explicit accumulator instead makes finite-width update behavior observable.
For example, Q15 dot with the currently executable i40 profile is written as:

```python
def q15_dot(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

Q15 FIR-sample preserves FIR intent while using the same explicit numeric
policy:

```python
def q15_fir(window: buffer[q15], coefficients: buffer[q15]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

The right operand of a fixed-point dot, FIR sample, or valid full-output FIR
may instead be embedded as compile-time raw values. Scalar reductions require
a static left-operand extent so the frontend can prove the lengths agree:

```python
def q15_fir_constexpr(
    window: buffer[q15, 5],
    coefficients: constexpr[q15] = [16384, -8192, 4096, -8192, 16384]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, wrap],
             rounding=nearest_even,
             overflow=saturate)
```

The constexpr parameter is not a runtime ABI argument. It lowers to a private
constant memref global, preserving immutable provenance for existing
proof-driven constant reduction and FIR specialization passes. Values must fit
the selected signed storage; constexpr is not yet a general expression
facility. A constexpr dot can therefore enter prefix-proof-authorized Vector
reassociation without turning a source assertion into legality authority.
For `fir_filter`, the compiler-owned global is exposed as a read-only tensor
value and removed again by bufferization. A static input extent is optional;
the coefficient count still determines each valid window, and the recovered
global provenance can authorize the existing prefix-proof Vector reduction.

The same fixed-point source forms accept `q31`. The executable Q31 profile
uses signed i32 storage with 31 fractional bits, an exact full product, and an
explicit i64/frac62 accumulator:

```python
def q31_dot(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
  return dot(lhs, rhs,
             accumulator=exact[64, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

`constexpr[q31]` reduction operands follow the same static-length rule and
must fit signed i32 storage. Raw-high Q31 products and implicit rescaling are
not part of the source profile.

Ordered f32 dot and FIR-sample kernels name their contraction policy instead
of a fixed-point accumulator and export policy:

```python
def f32_dot(lhs: buffer[f32], rhs: buffer[f32]) -> f32:
  return dot(lhs, rhs, contract=fma)
```

```python
def f32_fir(window: buffer[f32], coefficients: buffer[f32]) -> f32:
  return fir(window, coefficients, contract=off)
```

Full-output valid FIR uses rank-1 tensor values rather than mutable source
buffers. It returns a new tensor and preserves the existing `ondrix.fir_filter`
algorithm contract:

```python
def q15_fir_filter(input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients,
                    boundary=valid,
                    accumulator=exact[40, saturate],
                    rounding=nearest_even,
                    overflow=saturate)
```

Q31 uses the corresponding exact 64-bit accumulator profile. Fixed constexpr
coefficients are accepted in the same right-operand position. f32 replaces the
fixed-point policies with `contract=off|fma|fast`. Valid mode permits dynamic
or static tensor extents. A static result requires static input and coefficient
extents and must equal `input_length - coefficient_length + 1`; otherwise the
result extent must remain dynamic so the runtime shape contract is preserved.

`boundary=full` exposes the existing zero-padded full-output equation. Its
first source slice requires static input, coefficient, and result extents with
`result_length = input_length + coefficient_length - 1`; Sema rejects extent
overflow. The lowered fixed-point path keeps guarded ordered updates on both
padded edges and can use prefix-proof-authorized Vector reduction only in the
fully overlapping interior. Dynamic full output, same padding, stride,
dilation, streaming state, mutable destinations, and tensor indexing remain
available only through textual MLIR contracts.

Valid one-dimensional convolution and correlation use tensor values and the
same fixed-point or floating-point policy syntax as full-output FIR:

```python
def q15_convolution(
    input: tensor[q15,6], kernel: tensor[q15,3]) -> tensor[q15,4]:
  return convolution(input, kernel,
                     accumulator=exact[40,saturate],
                     rounding=nearest_even,
                     overflow=saturate)

def f32_correlation(
    input: tensor[f32,6], kernel: tensor[f32,3]) -> tensor[f32,4]:
  return correlation(input, kernel, contract=fma)
```

The fixed Q15 convolution/correlation forms may omit their policy when the
kernel extent is static. Export then uses the same nearest-even/saturating
language default as inferred Q15 dot/FIR.

Both forms require `result_length = input_length - kernel_length + 1`.
Correlation pairs increasing input and kernel indices. Convolution reverses
the kernel while preserving increasing input-window and accumulator-update
order. Dynamic tensor extents are accepted when the result is also dynamic.
Padding, stride, dilation, multidimensional/grouped forms, and constexpr
kernels are not part of this source slice.

Static packed-Q15 complex FFT uses an explicit experimental element spelling:

```python
def q15_cfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input)

def q15_icfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(input)
```

`complex_q15` is stored as one `i32` with the imaginary component in bits
31:16 and the real component in bits 15:0. The current builtins accept only
matching static extents of four or eight and emit closed radix-2 profiles:
`cfft` uses forward twiddles and `icfft` uses their conjugates. Both use exact
full complex products, nearest-even saturating Q30-to-Q15 product scaling,
and nearest-even saturating one-bit scaling at every butterfly stage. The
per-stage scaling makes the inverse profile include the `1/N` normalization.
This spelling does not imply a general source complex type; other sizes,
dynamic planning, and configurable complex policies remain unsupported.
After source generation, the opt-in
`--convert-ondrix-to-ondsp="vectorize-static-cfft"` mode maps independent
combine-stage butterflies to fixed-length Vector arithmetic while preserving
all stage and requantization boundaries.

The same closed packed-Q15 policy is available as one explicit two-result
butterfly:

```python
def q15_butterfly(
    a: complex_q15, b: complex_q15, twiddle: complex_q15)
    -> (complex_q15, complex_q15):
  return butterfly(a, b, twiddle)
```

Static-coefficient Q15 streaming FIR is the second bounded multi-result form:

```python
def q15_fir_stream(
    input: tensor[q15],
    coefficients: tensor[q15,3],
    state: tensor[q15,2])
    -> (tensor[q15], tensor[q15,2]):
  return fir_stream(input, coefficients, state)
```

The chunk extent may be dynamic. Coefficient, state, and next-state extents are
static, with `state_length = coefficient_length - 1`; output extent follows the
input chunk. Each output uses the same inferred exact Q15 accumulation as a
static-tap FIR sample, while next state is the chronological raw-sample suffix
and is not requantized. Whole and split chunks therefore have identical output
and final-state semantics.

These two forms do not introduce general tuples, assignments, configurable
complex arithmetic, recursive-state inference, or a stable C ABI.

One fixed direct-form-II SOS section is available through an explicit
recursive numeric profile:

```python
def q15_sos_df2_fixed(
    input: tensor[q15],
    coefficients: tensor[q15,1,5],
    scales: tensor[q15,1],
    state: tensor[q15,1,2])
    -> (tensor[q15], tensor[q15,1,2]):
  return sos_df2_fixed(
      input, coefficients, scales, state,
      accumulator=exact[40,saturate],
      state_rounding=nearest_even,
      state_overflow=saturate,
      output_rounding=toward_zero,
      output_overflow=wrap)
```

The coefficient row is `[b0,b1,b2,a1,a2]`; feedback is additive, so a
subtractive convention supplies negated `a1/a2`. State is `[d1,d2]`. The input
and output chunk extent may be dynamic, while coefficients `[1,5]`, scales
`[1]`, and state/next-state `[1,2]` are static. Unlike feed-forward `auto`
accumulation, all recursive update and quantization behavior is explicit.
Q31/f32 variants, additional direct forms, recursive Vector lowering, and a
stable state ABI are outside this source slice.

Compile it to textual MLIR with:

```sh
ondrix-compile input.ox -o output.mlir
```

The frontend expands `q15` and `q31` to their signed integer storage and
fractional positions, emits an exact full product, materializes either an
inferred exact Q15 accumulator or the declared i40/frac30 or i64/frac62
accumulator policy, and emits an explicit export.
Supported update and destination overflow modes are `wrap` and `saturate`.
Supported rounding modes are `toward_negative`, `toward_zero`, and
`nearest_even`.

f32 dot and FIR support `contract=off`, `contract=fma`, and `contract=fast`.
`off` preserves separate multiply and add operations, `fma` requires fused
updates, and `fast` explicitly permits fast-math transformations. Only `off`
and `fma` are bitwise differential contracts.

No target capability or physical register information enters source IR.
`llvm.emit_c_interface` marks the generated function for the existing AOT
pipeline, but the resulting C ABI is not stable. In particular, tensor results
currently use MLIR's bufferized ranked-memref descriptor convention in the test
wrapper. That convention is not part of `.ox` source semantics. Tensor
parameters are values; the frontend does not invent a `restrict` promise for
them.

This is not a general Python parser. Imports, classes, heap objects, arbitrary
expressions, and dynamic Python behavior are rejected. Scalar constants,
indexing, loops, mutable output buffers, multiple kernels, and inferred
accumulators outside the statically bounded Q15 real-reduction slice remain
unimplemented. Textual MLIR remains an independent and more complete compiler
entry point.
