def invalid_magnitude_result(spectrum: tensor[complex_q15,9]) -> tensor[q15,8]:
  return magnitude(spectrum)
