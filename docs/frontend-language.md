# Experimental `.ox` Frontend

`ondrix-compile` is a standalone C++ frontend. Its initial executable surface
accepts one or more `def`s per file, of which the last is the kernel the
module exports. Most current kernels use rank-1 buffers or tensors;
the fixed SOS slice also accepts the explicit rank-2 section layouts described
below. A `def` declares a DSP kernel entry point; it is not a general Python
function.

For a statically bounded Q15 dot, FIR sample, FIR decimation, FIR
interpolation, convolution, or correlation, omitting the accumulator policy
requests target-independent exact mathematical accumulation:

```python
def q15_fir_auto(
    window: buffer[q15, 4], coefficients: buffer[q15, 4]) -> q15:
  return fir(window, coefficients)
```

The frontend derives the smallest signed accumulator width that contains every
possible full-precision Q15 product sum, with a minimum width of 32 bits. Scalar
dot/FIR use the common operand extent; FIR decimation and
convolution/correlation use the coefficient or kernel extent for each output
window. The example above normalizes to an i34/frac30 accumulator, while a
three-tap correlation uses i33/frac30. This inferred width is not a hardware
register choice, and a target may not silently narrow it. Dynamic reductions
currently require an explicit finite profile because no finite exact width
follows from an unbounded runtime length.

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

The first source resampling slice exposes valid, phase-zero Q15 FIR decimation
by two:

```python
def q15_fir_decimate(
    input: tensor[q15,12], coefficients: tensor[q15,5]) -> tensor[q15,4]:
  return fir_decimate(input, coefficients, factor=2)
```

All extents are static and the result must have
`floor((input_length - coefficient_length) / 2) + 1` elements. Each output
uses the same increasing-tap ordered FIR equation and portable accumulator
inference as other static Q15 feed-forward reductions. It can lower through
the generic scalar path or through direct bufferization into ordered
fixed-width Vector products plus a scalar tail. Other factors, dynamic shapes,
nonzero phase, cross-output polyphase lowering, and Q31 resampling are not
part of this source slice.

Static phase-zero Q15 FIR interpolation by two is also available:

```python
def q15_fir_interpolate(
    input: tensor[q15,4], coefficients: tensor[q15,3]) -> tensor[q15,9]:
  return fir_interpolate(input, coefficients, factor=2)
```

The result extent must equal `(input_length - 1) * 2 + coefficient_length`.
Portable accumulator inference uses the maximum phase-tap bound,
`ceil(coefficient_length / 2)`, rather than counting inserted zeros. The
current source slice lowers through the generic scalar consumer;
dynamic shapes, constexpr coefficients, configurable phase/factor, stateful
interpolation, polyphase transforms, Vector, and target consumers remain
outside the source contract.

Both resampling builtins also accept the f32 profile, where the declared
contract replaces the accumulator and export policy:

```python
def f32_fir_decimate(
    input: tensor[f32,12], coefficients: tensor[f32,5]) -> tensor[f32,4]:
  return fir_decimate(input, coefficients, factor=2, contract=fma)
```

The index relations and extent rules are unchanged; only the per-output
reduction is contract indexed. Interpolation still skips the terms that would
multiply an inserted zero, which under f32 is a declared event rather than a
derivable rewrite. The two are not interchangeable even on all-finite data:
under `fma` an accumulator reaches `-0.0` when a real product underflows, and
materializing a following `+0.0 * c` with positive `c` rounds it back to
`+0.0`. A non-finite coefficient separates them too, since `0.0 * inf` and
`0.0 * NaN` are both NaN.

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

Static packed-Q15 FFT uses an explicit experimental complex element spelling:

```python
def q15_cfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return cfft(input)

def q15_icfft8(input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(input)

def q15_cfft_round_trip(
    input: tensor[complex_q15,8]) -> tensor[complex_q15,8]:
  return icfft(cfft(input))

def q15_rfft16(input: tensor[q15,16]) -> tensor[complex_q15,9]:
  return rfft(input)

def q15_irfft16(input: tensor[complex_q15,9]) -> tensor[q15,16]:
  return irfft(input)

def q15_rfft_round_trip(input: tensor[q15,16]) -> tensor[q15,16]:
  return irfft(rfft(input))
```

`complex_q15` is stored as one `i32` with the imaginary component in bits
31:16 and the real component in bits 15:0. The current builtins accept only
matching static extents of four or eight and emit closed radix-2 profiles:
`cfft` uses forward twiddles and `icfft` uses their conjugates. Both use
exact full complex products. The default stage policy is nearest-even
saturating Q30-to-Q15 product scaling and nearest-even saturating one-bit
scaling at every butterfly stage; optional `rounding=`/`overflow=`
parameters name one declared pair for both scale boundaries, admitting the
gated target-inventory combinations (`toward_negative` or
`nearest_ties_positive`, with `wrap` or `saturate`) — `nearest_even` with
`wrap` has no gate and is refused. The per-stage scaling makes the inverse
profile include the `1/N` normalization. This spelling does not imply a
general source complex type; other sizes and dynamic planning remain
unsupported.
`rfft` accepts static power-of-two real extents from 8 through 64 and returns
the compact natural Hermitian bins 0 through N/2, for `q15` and `q31`
elements alike. The Q15 `irfft` is narrower: it accepts only 5 or 9 packed
bins (real extents 8 or 16), so the larger Q15 `rfft` extents have no
`irfft` counterpart yet; the Q31 `irfft` accepts every bin count whose real
extent is a power of two in [8, 64]. DC and Nyquist imaginary components are
canonicalized to zero by the existing Ondrix contract.

`complex_q31` is the matching packed 32-bit spelling: one `i64` with the
imaginary component in bits 63:32 and the real component in bits 31:0. The
FFT-family builtins accept it — `cfft`/`icfft` at static power-of-two
extents from 4 through 64, `rfft` over `q31` reals, and `irfft` back to
them — and all emit the re-frozen packed-Q31 profile: raw-high per-term
products and toward_negative saturating stage scaling. That profile has
exactly one gated stage policy, so the optional `rounding=`/`overflow=`
parameters admit only `toward_negative` and `saturate`; the other real and
complex builtins remain Q15-only.

As a bounded expression-composition slice, the unary FFT-family builtins may
be nested when every intermediate type and extent satisfies the next
builtin's contract. Each nested call emits a separate Ondrix operation, so the
intermediate stage-scaling and requantization boundaries remain observable;
in particular, `irfft(rfft(input))` is not folded to an identity.

A kernel body may also name intermediate stages with local bindings before
its single return statement. Each local binds one builtin call, and every
later reference instantiates that call, so a local may be read any number of
times; a local no statement reads is a compile error. Reading a local twice
is not a second evaluation: the duplicated subtrees are `Pure` Ondrix
operations and the canonical pipeline's `canonicalize`/`cse` collapses them,
which is what lets an intermediate be reused, as in `add(mult(t, t), t)`.
Within this slice a `fir_filter` stage may feed the FFT chain — static Q15
tensors, the `valid` boundary, and the explicit executable accumulator
profile — and its coefficients may come from a runtime tensor parameter or
from a compile-time design expression. The available designs are `lowpass`,
which names the Hamming-windowed-sinc contract (`taps` odd in [3, 4095],
rational `cutoff` strictly inside (0, 1/2)), and the four window designs
`hamming(taps=N)`, `hann(taps=N)`, `blackman(taps=N)`, and
`kaiser(taps=N, beta=[num,den])`, each with `taps` in [2, 4096] and Kaiser's
rational `beta` in (0, 50]. Every design is evaluated at compile time under
the fail-closed quantization tie guard and is legal only in the coefficient
slot. The composed four-stage program is:

```python
def q15_filtered_spectrum(signal: tensor[q15,72]) -> tensor[q15,33]:
  taps = lowpass(taps=9, cutoff=[1,4])
  filtered = fir_filter(signal, taps, boundary=valid,
                        accumulator=exact[40,saturate],
                        rounding=nearest_even, overflow=saturate)
  spectrum = rfft(filtered)
  return magnitude(spectrum)
```

Other builtins still require direct parameter operands.
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
Supported rounding modes for this dot/FIR export policy are
`toward_negative`, `toward_zero`, `nearest_even`, and
`nearest_ties_positive`; a newly declared mode stays rejected here until
the corresponding contract opts in with its own object evidence. Individual
catalog builtins expose their own admissible choices at the call site
(for example `gain(..., rounding=...)` and the `root_rounding=` parameter
of `rms` and `magnitude`).

A fixed-point call site that names no contract past its algorithm parameters
takes the default: an accumulator inferred wide enough that no update can
wrap (so the update mode is vacuous), the `nearest_ties_positive` tie rule
that fixed-point export hardware realizes natively (add half, then shift;
`nearest_even` remains one declaration away for unbiased accumulation),
and saturation at the one boundary that does lose information. This covers
`dot`, `fir`, `fir_filter`, `fir_decimate`, `fir_interpolate`, `convolution`,
`correlation`, `fir_stream`, and `sos_df2_fixed`. Inference needs a static
tap count, so a dynamic coefficient extent is a diagnostic rather than a
guessed width. Any explicit parameter still overrides the default, and the
`.ox` object gate executes the inferred and the spelled contract against one
exact reference rather than asserting their equivalence.

`cic_decimate(x, stages=, rate=, delay=, state_overflow=, rounding=)` is the
one builtin with a mandatory contract parameter. Omission normally keeps a
contract default, but the cascade is only correct under `state_overflow=wrap`
and no reading of the source would justify choosing that silently, so leaving
it out is a parse error. `rounding=` keeps its usual optional form, since both
tie rules the boundary admits are defensible.

f32 dot, FIR, `matmul`, `rms`, `moving_average`, `dct`, `fir_decimate`,
`fir_interpolate`, `gain`, `lms`, and `goertzel` support `contract=off`,
`contract=fma`, and `contract=fast`; `sos_tdf2` admits only the two exact
modes, because its operation contract has no realization gate for `fast`.
The floating-point spellings name a contract where their
fixed-point counterparts name a rounding mode, because a floating-point result
has no requantization boundary to round; `rms(x, contract=...)` also accepts
any extent in range rather than only powers of two. f32 `rms` additionally
admits a scalar spelling — a static `buffer[f32, N]` operand returning a bare
`f32` — which lowers to `dot(x, x)` plus one division and one root, so the
result crosses the boundary without an owning allocation. `gain` is the one builtin
whose three declarations denote the same event graph, since a lone multiply
offers no tree to rebuild and no addend to fuse.

The f32 constants of `gain` and `lms` are spelled as rationals,
`gain(x, gain=[3, 8], contract=off)` and
`lms(x, d, w, step_size=[1, 16], contract=fma)`, because the lexer has no
floating-point literal. The numerator and denominator must not exceed `2^24`
and the denominator must be positive. That bound makes both operands exact in
binary32, so the division is performed in binary32 and the constant is the
single correctly rounded quotient of the exact rational. It also keeps every
admitted quotient normal, so no spelling can reach the subnormal range or
infinity. The `lms` step size must
additionally be non-negative, as on the fixed profile.
`off` preserves separate multiply and add operations in the stated order, and
`fma` pins each update as one explicit fused multiply-add event (a single
rounding). Both are exact contracts: every authorized schedule reproduces
their results bitwise for non-NaN outputs, and maps NaN outputs to NaN
outputs — the payload and sign of a NaN, and whether a signalling NaN is
quieted, are outside the contract, while signed zeros and infinities remain
bitwise. An output can be NaN even when no input is, so the qualification is
stated over outputs rather than inputs.

`contract=fast` names a set of permitted numeric rewrites rather than an
error bound: rebuilding the reduction's additive tree over the same indexed
terms — permutation as well as reparenthesization — and selection
of a fused multiply-add event at a permitted multiply-accumulate site.
Declaring it does not discharge the numeric obligation; it selects a
different one. A schedule is admissible under `fast` when its numeric event
graph is derivable from the source event graph using only those two rewrites
— terms are conserved, indices are covered exactly once, and no other
identity is used — and when the permissions the backend receives do not
exceed the declared set. `fast` does not assert that inputs are free of NaN
or infinity, does not abandon signed-zero observability, and does not permit
reciprocal division or approximate math functions; each of those would be a
separate declaration this language does not currently offer. A `fast` result
is therefore never compared bit for bit against a fixed expectation; the
implementation is characterized by an error budget measured against a
higher-precision reference, which is a quality measurement rather than a
guarantee the language makes.

### Transcendental Builtins

`log2(x)` and `exp2(x)` are inverses of each other over static rank-1 Q15
tensors. The source type system names only the i16 storage, so the two
readings their contract distinguishes — the unsigned `Q0.16` magnitude and
the signed `Q5.11` exponent — are supplied by the binding rather than spelled
at the call site, the same projection `dct`'s derived output fraction uses.
Neither is part of the composable set yet, so like `dct`, `rms`, `sine`, and
`cosine` they take an operand name rather than a nested expression.

`phase(z)` maps a `complex_q15` tensor to the unsigned turn its elements'
arguments name, in the same reading `sine` consumes. It composes exactly
where `magnitude` does — `phase(rfft(x))` is the phase spectrum. Its contract
admits exactly one tie rule, so unlike `magnitude`'s `root_rounding=` there
is no rounding parameter to accept, and spelling one is a parse error.

### Elementwise Builtins

`add`, `sub`, `mult`, `abs`, `negate`, `offset`, and `shift` map static rank-1
Q15 tensors elementwise:

```python
def q15_envelope(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return add(mult(x, y), shift(abs(sub(x, y)), amount=-2), overflow=wrap)
```

Both boundary parameters are optional and both take the language default,
`rounding=nearest_ties_positive` and `overflow=saturate`. `offset` names a raw Q1.15
`bias` and `shift` a signed `amount` in `[-15, 15]`; a left shift declares a
tie rule too, even though the amount makes it vacuous, so changing the amount
never silently changes which rule applies.

The family is fixed point only. An elementwise IEEE operation has no
requantization boundary, so an f32 profile would add source surface without
adding any contract to declare.

These builtins are part of the composable set — the FFT family, `magnitude`,
a `fir_filter` input stage, and the elementwise members — which is closed
under nesting: any member may stand where an operand name may. A parameter
may be read more than once, since a tensor operand is a value, so `mult(x, x)`
is a squaring kernel rather than an aliasing question.

### Single-Bin And Recursive Builtins

`goertzel(x, bin=, contract=)` returns the squared magnitude of one DFT bin as
a single-element tensor, over the quantized recursion the operation contract
states. Only the f32 profile has a source spelling: the Q15 energy is
`tensor<1xi64>` and no source type names that storage width. The bin must lie
in `[0, N/2]` and the input extent in `[2, 4096]`.

`sos_tdf2` is the f32 transposed-direct-form-II cascade and the second
recursive source binding beside the fixed `sos_df2_fixed`:

```python
def f32_sos_tdf2(
    input: tensor[f32],
    coefficients: tensor[f32,2,5],
    scales: tensor[f32,2],
    state: tensor[f32,2,2])
    -> (tensor[f32], tensor[f32,2,2]):
  return sos_tdf2(input, coefficients, scales, state, contract=off)
```

Coefficients are `[sections, 5]` in the order `[b0, b1, b2, a1, a2]`, scales
are `[sections]`, and state is `[sections, 2]` in `[z1, z2]` order; the
section count is static and at least one. Denominator feedback is added, so a
subtractive source convention supplies negated `a1`/`a2`. Empty input returns
an empty output and the unchanged state.

### Named Functions

A file may declare several functions. The last one is the kernel the module
exports; each earlier one is a named body that a later function may call:

```python
def window(x: tensor[q15,71], c: tensor[q15,8]) -> tensor[q15,64]:
  return fir_filter(x, c, boundary=valid)

def spectrum(x: tensor[q15,71], c: tensor[q15,8]) -> tensor[q15,33]:
  return magnitude(rfft(window(x, c)))
```

A call is instantiated where it appears: the callee's body replaces the call
and the arguments replace the callee's parameter references. Every contract
written in the callee is copied unchanged, which is what makes the contract
travel with the name rather than being re-declared at each call site.

Each callee is checked once against its own signature, so a body that cannot
hold its declaration is reported at the callee. A call site then only has to
match its arguments to that signature: arity, and for an argument naming one
of the caller's parameters, the source type, container kind, and shape. The
declared result type therefore holds at the call without the caller
re-deriving it.

Arguments are parameter or local names. A callee is visible only to functions
declared after it, so recursion cannot be written, and a function calling
itself is a diagnostic rather than a missing-name error. A call may appear
anywhere a nested expression already may — the whole return expression, a
local binding, or a stage of an FFT chain — and single-result functions only.

Only the exported kernel becomes a `func.func`. Calls are not a source-level
ABI, and no calling convention is implied: a stable public kernel ABI is still
future work.

No target capability or physical register information enters source IR.
`llvm.emit_c_interface` marks the generated function for the existing AOT
pipeline, but the resulting C ABI is not stable. In particular, tensor results
currently use MLIR's bufferized ranked-memref descriptor convention in the test
wrapper. That convention is not part of `.ox` source semantics. Tensor
parameters are values, so aliasing is not observable in the source language at
all; the storage question only appears at the emitted ABI, and there it is a
DECLARED precondition rather than an inference: a caller of the bufferized
entry point must pass storage for each parameter and result that is disjoint
from every other parameter's and result's storage. Distinct mutable `buffer`
parameters of one kernel carry the same declared non-overlap precondition at
the language level. Schedule transforms that reorder loads against stores rely
on exactly this contract for function entry arguments — everything else they
prove or refuse statically — so the precondition is part of the language and
ABI surface, never a hidden assumption of one pass.

This is not a general Python parser. Imports, classes, heap objects, arbitrary
expressions, and dynamic Python behavior are rejected. Scalar constants,
indexing, loops, mutable output buffers, multiple exported kernels, and
inferred accumulators outside the statically bounded Q15 real-reduction slice
remain unimplemented. Textual MLIR remains an independent and more complete compiler
entry point.

## Semantic Boundary And Design Rules

This document defines the intended boundary between the experimental `.ox`
source language, project dialects, and upstream MLIR dialects. The implemented
parser currently covers Q15/Q31/f32 real reductions, bounded packed-Q15
complex transforms, explicit-state Q15 streaming FIR, and one static-section
Q15 fixed SOS profile. Fixed-point reductions may use compiler-owned constexpr
coefficients. The broader items below remain design constraints.

### Design Rule

The frontend must not mirror every source construct into a custom operation.
A custom project IR construct is justified only when it preserves algorithm,
numeric, or target-capability information that upstream MLIR cannot recover
after lowering.

Source constructs map to owning dialects as follows:

| Source concept | MLIR ownership |
| --- | --- |
| Functions, calls, and returns | `func` |
| Structured control flow | `scf` and `cf` |
| Ordinary integer and floating-point arithmetic | `arith` and `math` |
| Buffers, views, and subviews | `memref` or `tensor` |
| Semantic lane vectors | builtin `vector` types and the Vector dialect |
| DSP algorithm intent | Ondrix dialect |
| Fixed-point products, accumulators, and export policy | Ondsp dialect |
| Proven target capabilities | OrtumCore dialect |

Physical registers, register tuples, encodings, scheduling, and instruction
selection remain private-backend concerns.

### Source-Level Algorithm Constructs

The stable source surface should remain smaller than the internal compiler
surface. Candidate source constructs include FIR, convolution/correlation,
FFT-family transforms, recursive filters, and resampling. Each construct must
define its output domain, indexing, state, and numeric boundaries before it is
bound to a stable Ondrix operation.

The current operations have different expected roles:

- `ondrix.fir` is an algorithm operation for one pre-windowed output sample.
- `ondrix.dot` may be frontend convenience syntax. Without axis, conjugation,
  batching, masking, or other retained intent, it may lower immediately to
  `ondsp.reduce_mac`.
- `ondrix.butterfly` is expected to be produced by transform decomposition;
  it need not be a user-facing source primitive.
- fixed-point casts and quantization are source expressions whose normalized
  semantics belong to Ondsp rather than the algorithm dialect.

Full-output and streaming FIR use distinct contracts for boundaries, output
shape, state, and aliasing. The experimental `ondrix.fir_filter` and
`ondrix.fir_stream` operations encode those separate contracts. The first
source binding exposes streaming as an explicit `(output, next_state)` result
and must not infer it from the single-sample operation.

The experimental `ondrix.sos_filter_tdf2` operation is the first recursive
filter contract. It deliberately names one form and fixes coefficient order,
feedback sign convention, per-section scaling, state layout, update order, and
floating-point contraction. A future source-level `iir` construct must choose
or infer those semantics explicitly; it must not treat all direct forms or
fixed-point state-quantization schemes as interchangeable.

### Source Types and Numeric Policy

The source language should provide an explicit fixed-point value type, with
convenience aliases such as Q15 and Q31. The unambiguous form records:

```text
fixed<signedness, storage-bits, fractional-bits>
```

Source syntax should normally avoid exposing normalized MLIR types such as
`!ondsp.acc`. High-level kernels instead choose one of two accumulator modes:

- `auto`: overflow is not part of the source behavior. The compiler must infer
  a sufficient width or reject the program when it cannot prove one.
- `exact`: width and per-update overflow are observable program semantics,
  suitable for bit-exact compatibility with an existing DSP kernel.

The normalized Ondsp IR always materializes the selected width, fractional
position, signedness, and update overflow. A target must not silently replace
`auto` exact accumulation with a narrower saturating capability.

The first implemented `auto` slice covers statically bounded signed-Q15 dot,
FIR-sample, FIR-decimation, convolution, and correlation reductions. It
derives a full-product accumulator width from the complete signed input domain
and the number of products in each output, and rejects unbounded lengths. Other
algorithms must provide their own sound range/state analysis before adopting
`auto`; Q-format alone never selects a target accumulator.

Q15 `fir_stream` reuses this rule only because every output remains a
fixed-tap feed-forward reduction. Its chronological next-state tensor consists
of source samples and does not pass through the accumulator/export lifecycle.
This does not authorize inferred accumulator or state widths for SOS/IIR
recurrences.

The first source-level fixed SOS binding therefore requires an explicit
`exact[40, ...]` accumulator and separate state/output rounding and overflow
policies. Its coefficient, scale, and state section layout is static; the
frontend does not infer recursive range bounds or reinterpret Q15 as a target
accumulator choice.

Leaving an accumulator requires explicit destination format, rounding, and
overflow policy. Source defaults, if introduced, must be language defaults and
must not depend on the selected target.

### Compile-Time Algorithm Facts

The frontend should communicate facts through ordinary source semantics:

- immutable coefficients use a `const` declaration;
- windows and complete signals use distinct checked types or expressions;
- static extents remain compile-time shape information;
- stateful kernels expose their state and update order.

Properties such as zero taps, sparsity, symmetry, antisymmetry, repeated
coefficients, powers of two, and trivial twiddles are inferred from immutable
values. A user-authored promise must not enable a rewrite unless it is verified
or represented by an explicit checked assumption.

### Optimization Contract

The intended pipeline is:

```text
.ox parser and AST
  -> source type, shape, and numeric checking
  -> Ondrix algorithm intent plus upstream MLIR
  -> algorithm fact inference and transforms
  -> normalized Ondsp numeric semantics
  -> scalar, Vector, or OrtumCore consumer
```

An algorithm transform first identifies a mathematical identity; Ondsp then
classifies that identity under the concrete product, accumulator, rounding,
and overflow contract, using the equivalence taxonomy and proof-fact rules
of the Transform Equivalence section in `fixed-point-semantics.md`.
Optimized variants should normally be represented with the same
algorithm operations, normalized Ondsp operations, and upstream dialects,
rather than one custom operation per theorem or fusion pattern.

### Operation Admission Checklist

A new stable Ondrix operation requires all of the following:

1. A defined source-level construct or compiler-generated algorithm role.
2. Information that cannot be recovered from upstream MLIR after lowering.
3. Complete shape, state, memory, and numeric verifier rules.
4. A public generic lowering or executable consumer.
5. At least one real analysis, transform, or target consumer.

Operations that do not satisfy these conditions remain experimental or are
lowered directly to Ondsp/upstream MLIR.
