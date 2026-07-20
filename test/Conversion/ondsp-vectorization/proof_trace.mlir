// RUN: ondrix-opt %s --vectorize-ondsp-constant-saturating-memref-reduce="vector-width=4 max-elements=8 proof-trace-output=%t.json" | FileCheck %s --check-prefix=TRANSFORMED
// RUN: FileCheck %s --input-file=%t.json --check-prefix=TRACE
// RUN: ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.json max-elements=8" | FileCheck %s --check-prefix=ORIGINAL
// RUN: sed 's/"value":"-8"/"value":"-7"/' %t.json > %t.tampered.json
// RUN: not ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.tampered.json max-elements=8" 2>&1 | FileCheck %s --check-prefix=TAMPERED
// RUN: sed 's/"subject_ordinal":1/"subject_ordinal":0/' %t.json > %t.duplicate.json
// RUN: not ondrix-opt %s --verify-ondsp-constant-reassociation-proof-trace="proof-trace-input=%t.duplicate.json max-elements=8" 2>&1 | FileCheck %s --check-prefix=DUPLICATE

memref.global "private" constant @coefficients : memref<8xi16> =
  dense<[1, -2, 3, -4, 5, -6, 7, -8]>

func.func @safe(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @coefficients : memref<8xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

func.func @safe_second(%input: memref<8xi16>)
    -> !ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate> {
  %coefficients = memref.get_global @coefficients : memref<8xi16>
  %zero = ondsp.acc_zero
      : !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  %result = ondsp.reduce_mac %zero, %input, %coefficients {
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>
  } : (!ondsp.acc<storage = i40, frac = 30, signed,
                  update_overflow = saturate>,
       memref<8xi16>, memref<8xi16>)
      -> !ondsp.acc<storage = i40, frac = 30, signed,
                    update_overflow = saturate>
  return %result : !ondsp.acc<storage = i40, frac = 30, signed,
                              update_overflow = saturate>
}

// TRANSFORMED-LABEL: func.func @safe
// TRANSFORMED: vector.reduction <add>
// TRANSFORMED: ondsp.acc_add_term
// TRANSFORMED-NOT: ondsp.reduce_mac

// ORIGINAL-LABEL: func.func @safe
// ORIGINAL: ondsp.reduce_mac
// ORIGINAL-LABEL: func.func @safe_second
// ORIGINAL: ondsp.reduce_mac

// TRACE-DAG: "schema_version":{{ *}}1
// TRACE-DAG: "vector_width":{{ *}}4
// TRACE-DAG: "analysis_max_elements":{{ *}}8
// TRACE-DAG: "candidate_reduction_count":{{ *}}2
// TRACE-DAG: "kind":"no_overflow_chunk_reassociation"
// TRACE-DAG: "subject_ordinal":{{ *}}0
// TRACE-DAG: "subject_ordinal":{{ *}}1
// TRACE-DAG: "numeric_storage_width":{{ *}}16
// TRACE-DAG: "accumulator_storage_width":{{ *}}40
// TRACE-DAG: "chunk_width":{{ *}}4
// TRACE-DAG: "coefficients":[
// TRACE-DAG: "original_prefixes":[
// TRACE-DAG: "reassociated_prefixes":[

// TAMPERED: error: proof trace record no longer matches this reduction: {{[01]}}
// DUPLICATE: error: invalid or duplicate proof trace record 1
