def invalid_local_collision(signal: tensor[q15,16]) -> tensor[complex_q15,9]:
  signal = rfft(signal)
  return rfft(signal)
