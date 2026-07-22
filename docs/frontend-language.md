# Experimental `.ox` Frontend

`ondrix-compile` is a standalone C++ frontend. Its first executable slice intentionally
accepts one kernel per file and one dynamic rank-1 signed-Q15 dot expression:

```python
kernel q15_dot(lhs: buffer[q15], rhs: buffer[q15]) -> q15:
  return dot(lhs, rhs,
             accumulator=exact[40, saturate],
             rounding=nearest_even,
             overflow=saturate)
```

Compile it to textual MLIR with:

```sh
ondrix-compile input.ox -o output.mlir
```

The frontend expands `q15` to signed `i16` storage with 15 fractional bits,
emits an exact full product, materializes the declared signed i40/frac30
accumulator policy, and emits an explicit Q15 export. Supported update and
destination overflow modes are `wrap` and `saturate`. Supported rounding modes
are `toward_negative`, `toward_zero`, and `nearest_even`.

No target capability or physical register information enters source IR.
`llvm.emit_c_interface` marks the generated function for the existing AOT
pipeline, but the resulting C ABI is not stable.

This is not a general Python parser. Functions declared with `def`, imports,
classes, heap objects, arbitrary expressions, and dynamic Python behavior are
rejected. f32, FIR, constants, indexing, loops, output buffers, multiple
kernels, and inferred accumulators remain unimplemented. Textual MLIR remains
an independent and more complete compiler entry point.
