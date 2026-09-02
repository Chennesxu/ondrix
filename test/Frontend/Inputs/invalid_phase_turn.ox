def invalid_phase_turn(spectrum: tensor[complex_q15,9]) -> tensor[q15,9]:
  return phase(spectrum, turn=q7)
