def invalid_sine_extent(phase: tensor[q15,8192]) -> tensor[q15,8192]:
  return sine(phase)
