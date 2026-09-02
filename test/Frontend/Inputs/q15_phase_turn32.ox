def q15_phase_turn32(spectrum: tensor[complex_q15,9]) -> tensor[q31,9]:
  return phase(spectrum, turn=q31)
