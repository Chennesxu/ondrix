// RUN: not ondrix-opt %s -split-input-file --convert-ondsp-to-ortumcore 2>&1 | FileCheck %s

module attributes {
  test.acc_type = !ondsp.acc<storage = i40, frac = 30, signed>
} {
  // CHECK: attribute 'test.acc_type' contains source accumulator type '!ondsp.acc<storage = i40, frac = 30, signed>'
  func.func @module_type_attribute() {
    return
  }
}

// -----

module {
  func.func @unsupported_function_metadata_attribute() attributes {
    test.acc_type = !ondsp.acc<storage = i40, frac = 29, signed>
  } {
    // CHECK: attribute 'test.acc_type' contains source accumulator type '!ondsp.acc<storage = i40, frac = 29, signed>'
    return
  }
}

// -----

module attributes {
  test.nested_types = [{acc = [!ondsp.acc<storage = i40, frac = 30, signed>]}]
} {
  // CHECK: attribute 'test.nested_types' contains source accumulator type '!ondsp.acc<storage = i40, frac = 30, signed>'
  func.func @nested_accumulator_attribute() {
    return
  }
}
