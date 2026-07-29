def q15_magnitude_bad(spectrum: tensor[complex_q15,9]) -> tensor[q15,9]:
  return magnitude(spectrum, root_rounding=toward_zero)
