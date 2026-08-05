def invalid_phase_rounding(spectrum: tensor[complex_q15,9]) -> tensor[q15,9]:
  return phase(spectrum, rounding=toward_negative)
