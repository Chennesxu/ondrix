def invalid_sos_tdf2_q15(
    input: tensor[q15,8],
    coefficients: tensor[q15,1,5],
    scales: tensor[q15,1],
    state: tensor[q15,1,2])
    -> (tensor[q15,8], tensor[q15,1,2]):
  return sos_tdf2(input, coefficients, scales, state, contract=off)
