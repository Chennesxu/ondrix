def f32_magnitude_spectrum(input: tensor[f32, 64]) -> tensor[f32, 33]:
  return magnitude(rfft(input, contract = fma), contract = fma)
