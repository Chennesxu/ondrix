def f32_phase_spectrum(input: tensor[f32, 64]) -> tensor[f32, 33]:
  return phase(rfft(input, contract = off), contract = off)
