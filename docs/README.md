# ondrix Documentation

This directory contains public documentation for implemented or
stable-candidate Ondrix behavior, organized along the architecture axes
rather than per operation:

- [Current implementation status](status.md) — the capability matrix; one
  row per algorithm family with its evidence state.
- [Fixed-point semantics](fixed-point-semantics.md) — the target-independent
  Ondsp numeric model: products, accumulators, rounding, overflow, and
  reduction legality.
- [f32 contract evidence ledger](f32-contract-evidence.md) — per operation and
  contract: which permissions are spent, which are emitted, and which
  declarations have no executed evidence behind them.
- [Experimental `.ox` frontend](frontend-language.md) — the source language
  reference and the frontend semantic boundary and design rules.

The authoritative contract for each individual operation is its description
in the dialect definition (`include/ondrix/Dialect/*/IR/*.td`), which lives
next to the verifier that enforces it and is reviewed with the code. This
directory documents only cross-cutting semantics.

Internal plans, private target material, review notes, and task history are
not part of this directory.
