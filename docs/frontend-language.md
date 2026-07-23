# Experimental `.ox` Frontend

`ondrix-compile` is a standalone C++ frontend. Its initial executable surface
accepts one kernel per file and dynamic rank-1 buffers. Q15 dot is written as:

```python
kernel q15_dot(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

Q15 FIR-sample preserves FIR intent while using the same numeric policy:

```python
kernel q15_fir(window: buffer[q15], coefficients: buffer[q15]) -> q15:
  return fir(window, coefficients,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

The right operand of a fixed-point dot or FIR may instead be embedded as
compile-time raw values. A static left-operand extent is required so the
frontend can prove the reduction lengths agree:

```python
kernel q15_fir_constexpr(
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

The same fixed-point source forms accept `q31`. The executable Q31 profile
uses signed i32 storage with 31 fractional bits, an exact full product, and an
explicit i64/frac62 accumulator:

```python
kernel q31_dot(lhs: buffer[q31], rhs: buffer[q31]) -> q31:
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
kernel f32_dot(lhs: buffer[f32], rhs: buffer[f32]) -> f32:
  return dot(lhs, rhs, contract=fma)
```

```python
kernel f32_fir(window: buffer[f32], coefficients: buffer[f32]) -> f32:
  return fir(window, coefficients, contract=off)
```

Full-output valid FIR uses rank-1 tensor values rather than mutable source
buffers. It returns a new tensor and preserves the existing `ondrix.fir_filter`
algorithm contract:

```python
kernel q15_fir_filter(input: tensor[q15], coefficients: tensor[q15]) -> tensor[q15]:
  return fir_filter(input, coefficients,
                    boundary=valid,
                    accumulator=exact[40, saturate],
                    rounding=nearest_even,
                    overflow=saturate)
```

Q31 uses the corresponding exact 64-bit accumulator profile. f32 replaces the
fixed-point policies with `contract=off|fma|fast`. Static tensor extents are
optional; when all three extents are static, the result must equal
`input_length - coefficient_length + 1`. Only `boundary=valid` is currently a
source feature. Full padding, streaming state, mutable destinations, and
tensor indexing remain available only through textual MLIR contracts.

Compile it to textual MLIR with:

```sh
ondrix-compile input.ox -o output.mlir
```

The frontend expands `q15` and `q31` to their signed integer storage and
fractional positions, emits an exact full product, materializes the declared
i40/frac30 or i64/frac62 accumulator policy, and emits an explicit export.
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

This is not a general Python parser. Functions declared with `def`, imports,
classes, heap objects, arbitrary expressions, and dynamic Python behavior are
rejected. Scalar constants, indexing, loops, mutable output buffers, multiple
kernels, and inferred accumulators remain unimplemented. Textual MLIR remains an
independent and more complete compiler entry point.
