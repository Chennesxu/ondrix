# Streaming FIR Contract

`ondrix.fir_stream` is the first explicit stateful algorithm contract. It
processes one rank-1 input chunk and returns both an equal-length output chunk
and the history required by the next invocation. The operation is tensor-only,
so state ownership is represented by SSA values rather than hidden mutation or
an aliasing promise.

For coefficient length `K`, the state length is exactly `K - 1`. State samples
are ordered from oldest to newest. Given:

```text
extended = concat(state, input)
```

the operation computes:

```text
output[n] = ordered_fir_sample(extended[n : n + K], coefficients)
next_state = suffix(extended, K - 1)
```

Every output starts from numeric zero and visits taps in increasing index
order. An empty input produces an empty output and preserves state. `K = 1`
uses an empty state. Chunks shorter than the state are valid: the next state
contains the required suffix of the previous state followed by the new input.

## Numeric Contract

Fixed-point streams reuse the same explicit Ondsp lifecycle as full-output FIR:

- input, coefficient, and state tensors use the source fixed storage type;
- each output uses the declared product and accumulator update policy;
- output conversion explicitly declares destination format, rounding, and
  destination overflow;
- next state retains source sample storage and does not pass through the output
  requantization policy.

Floating-point streams use the declared `off`, `fma`, or `fast` contract for
each increasing-index update. Chunk boundaries do not authorize tap
reassociation.

## Executable Slice

The generic lowering creates ordered SCF loops over output samples and taps,
selecting each sample from state or input according to its index in
`extended`. A second loop constructs the next state suffix. Dynamic execution
checks require non-empty coefficients, `state_length + 1 == K`, and an
indexable combined state/input extent. When a result carries a static extent
but its corresponding input or state extent is dynamic, lowering also checks
that the runtime extent matches before constructing or writing that result.

Tracked object-and-C execution covers signed Q15, signed Q31, f32 FMA, and an
f32 `off` case that distinguishes separate multiply/add from contraction. The
same input is evaluated both as one chunk and as multiple consecutive chunks;
all outputs and final state are compared against independent C references.
Tests also cover an initial chunk shorter than state, an empty chunk, `K = 1`,
dynamic extents, accumulator/export behavior, allocation release, and
process-level rejection of invalid state, coefficient, and result extents.

## Deliberate Limits

The first implementation is an ordered generic scalar path. It does not yet
define:

- in-place public buffer semantics or a stable C ABI;
- output-axis or cross-chunk Vector scheduling;
- state reset, decimation/interpolation, stride, or dilation;
- concurrent ownership of one state value by multiple stream invocations;
- target capability selection for state movement or circular buffers.

These require separate legality and ownership decisions. In particular, a
private backend may implement the state with a circular buffer or register
window, but those physical choices must not change the public chronological
state value.
