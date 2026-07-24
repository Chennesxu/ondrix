def q15_butterfly(
    a: complex_q15, b: complex_q15, twiddle: complex_q15)
    -> (complex_q15, complex_q15):
  return butterfly(a, b, twiddle)
