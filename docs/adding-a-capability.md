# Adding a Target Capability: the Proof Obligations

A target capability is not an operation definition — it is a set of proof
obligations that all have to exist before the capability may participate in
compilation. This checklist is the complete set, in the order they are
normally discharged. The dual-lane MAC (`ortumcore.dmac`) is the most recent
capability to walk every step and serves as the reference example throughout.

A capability is *incomplete* until every box is checked; an incomplete
capability must not be reachable from any conversion, so that the fail-closed
walls still reject programs that would need it.

## 1. The contract

- [ ] **ODS definition** with a `description` that states the exact numeric
  equation, what the operation deliberately does NOT include (export,
  requantization), and the claim that makes it a capability rather than a
  convenience (for `dmac`: the *pairing* is the capability; the lanes never
  interact). `include/ondrix/Dialect/ortumcore/IR/OrtumCoreOps.td`.
- [ ] **Verifier** for whatever the type system cannot state. Prefer types
  and `AllTypesMatch`-style traits over verifier prose.

## 2. Target admission

- [ ] **Profile method** on `OrtumCoreTargetProfile` admitting exactly the
  proven domain (storage, fraction, signedness, overflow, shift ranges).
  The profile is the single authority; conversions ask it, they do not
  re-derive it.

## 3. The producer (proven conversion)

- [ ] **Conversion pattern** from the `ondsp` policy level, admitted only
  where the profile says yes, with a diagnostic naming the failed
  obligation otherwise. If the realization is a *composition* (like the
  ties-positive export), its exactness argument goes in the pass
  `description` in `Passes.td` — that prose is the canonical home of the
  soundness argument and is reviewed with the code.
- [ ] **Fail-closed check**: a program using the neighboring, unproven
  shape must be *rejected*, not approximated. Add the negative conversion
  witness that proves it.

## 4. The generic consumer (emulation)

- [ ] **Emulation expansion** in `convert-ortumcore-to-ondsp-emulation`,
  expressing the capability exactly in `ondsp` events (for `dmac`:
  literally two `ondsp.mac`). This keeps every capability executable on any
  host and anchors the differential legs.
- [ ] **Expansion witness**: conversion test showing the expansion and
  `CHECK-NOT`-ing target-dialect residue.

## 5. The differential gates

- [ ] **Object+C execution gate** (`test/Execution/`): the capability
  compiled through the emulation path against an independently formulated C
  reference — different arithmetic formulation, not a re-export of the
  compiler's helpers.
- [ ] **Discriminating witnesses**, named, each chosen so the *rejected*
  semantics gives a different answer (rails for saturation-vs-wrap,
  negative halves for tie rules, asymmetric lanes for crosstalk). Where
  cheap, add a self-check that the corpus still discriminates.
- [ ] **Dialect roundtrip test** for the printed form.

## 6. The records

- [ ] **`docs/status.md` row** stating the capability, its evidence level
  (see `research-claims.md`), and what remains unsupported.
- [ ] If the capability changes what the default pipeline can do, update
  the affected pass descriptions rather than duplicating them in docs.

## Order matters

Steps 1–2 before 3: a conversion must have a profile to ask. Step 4 before
5: the execution gate runs through the emulation. Nothing ships with step 5
missing — a capability without a differential gate is a claim without
evidence, and the gates in this project have repeatedly found real defects
that every earlier step passed.
