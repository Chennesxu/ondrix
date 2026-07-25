def invalid_fir_stream_state(
    input: tensor[q15],
    coefficients: tensor[q15,3],
    state: tensor[q15,3])
    -> (tensor[q15], tensor[q15,2]):
  return fir_stream(input, coefficients, state)
