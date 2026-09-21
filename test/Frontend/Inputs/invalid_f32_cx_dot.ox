def invalid_f32_cx_dot(x: buffer[complex_f32, 8], y: buffer[complex_f32, 8]) -> complex_f32:
  return cx_dot(x, y, contract=off)
