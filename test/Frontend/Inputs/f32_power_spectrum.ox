def f32_power_spectrum(input: tensor[f32, 64]) -> tensor[f32, 33]:
  return power(rfft(input, contract = off), contract = off)
