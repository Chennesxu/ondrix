#ifndef ONDRIX_DIALECT_ONDSP_IR_ONDSPTYPES_TD
#define ONDRIX_DIALECT_ONDSP_IR_ONDSPTYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "ondrix/Dialect/ondsp/IR/OndspDialect.td"
include "ondrix/Dialect/ondsp/IR/OndspEnums.td"

class Ondsp_Type<string name, string typeMnemonic>
    : TypeDef<Ondsp_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Ondsp_AccType : Ondsp_Type<"Acc", "acc"> {
  let summary = "DSP accumulator state with explicit update semantics";
  let description = [{
    Carries the raw storage width, binary-point position, signedness, and the
    overflow behavior applied after every accumulator update. Update overflow
    is distinct from the destination overflow policy used by `acc_export`.

    `lanes` declares how many independent accumulators the value carries. It
    defaults to one and is printed only when greater than one, so single-lane
    assembly is unchanged. Every lane has the identical declared semantics and
    no operation ever mixes lanes: a multi-lane accumulator is `lanes` separate
    ordered folds carried together, never a partial sum awaiting a horizontal
    reduction. That is what makes cross-output batching an order-preserving
    rewrite rather than a reassociation. Only `acc_zero`, `mac`, and
    `acc_export` accept more than one lane; every other accumulator consumer
    fails closed.
  }];
  let parameters = (ins
    "mlir::Type":$storage,
    "unsigned":$frac,
    "Signedness":$signedness,
    "OverflowMode":$updateOverflow,
    DefaultValuedParameter<"unsigned", "1">:$lanes
  );
  let assemblyFormat =
      "`<` `storage` `=` $storage `,` `frac` `=` $frac `,` $signedness `,` "
      "`update_overflow` `=` $updateOverflow (`,` `lanes` `=` $lanes^)? `>`";
  let builders = [
    // Builds the single-lane accumulator. Producers that do not batch outputs
    // ask for exactly this, which is why the short form is the defaulted one:
    // declaring more lanes is a deliberate act, and only `acc_zero`, `mac`, and
    // `acc_export` accept the result.
    TypeBuilder<(ins "mlir::Type":$storage, "unsigned":$frac,
                     "Signedness":$signedness, "OverflowMode":$updateOverflow), [{
      return $_get($_ctxt, storage, frac, signedness, updateOverflow, /*lanes=*/1);
    }]>,
  ];
  let genVerifyDecl = 1;
}

def Ondsp_Acc : Type<
    CPred<"::llvm::isa<::ondrix::ondsp::AccType>($_self)">,
    "ondsp accumulator type", "::ondrix::ondsp::AccType">;

#endif
