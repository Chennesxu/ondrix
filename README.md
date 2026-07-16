# ondrix

ondrix is a language and compiler framework for portable embedded DSP kernels.
It separates algorithm intent, target-independent numeric semantics, and
target capabilities in a layered MLIR stack, with AOT code generation as the
primary execution model.

The intended compilation flow is:

```text
ondrix language
  -> ondrix dialect
  -> ondsp dialect
  -> generic scalar, generic Vector, or ortumcore path
  -> LLVM dialect / LLVM IR
  -> object file or firmware integration
```

The three project dialects have distinct responsibilities:

- `ondrix` preserves DSP algorithm and kernel intent.
- `ondsp` defines target-independent numeric and accumulator semantics.
- `ortumcore` represents abstract target capabilities without exposing
  private instruction encodings or physical registers.

Textual MLIR is currently the supported development entry point. The repository
includes executable signed-Q15 scalar and fixed-width Vector AOT paths for
single-output FIR and dot-product kernels. The source language and additional
target consumers remain under development.

## Documentation

- [Documentation index](docs/README.md)
- [Current implementation status](docs/status.md)
- [Executable signed-Q15 semantics](docs/q15-semantics.md)

## Build

ondrix tracks LLVM/MLIR 17.0.6 at commit
`6009708b4367171ccdbf4b5905cb6a803753fe18`. This is the tested baseline.
Other LLVM 17 revisions are accepted with a configuration warning for
migration experiments; other major versions are rejected.

Configure against an external LLVM/MLIR build:

```sh
cmake -G Ninja \
  -S /path/to/ondrix \
  -B /path/to/ondrix/build \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  -DLLVM_EXTERNAL_LIT=/path/to/llvm-project/build/bin/llvm-lit
```

Build and run the regression suite:

```sh
ninja -C /path/to/ondrix/build ondrix-opt
ninja -C /path/to/ondrix/build check-ondrix
```

## License

See [LICENSE](LICENSE).
