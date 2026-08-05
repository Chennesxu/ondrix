def invalid_sos_tdf2_fast(
    input: tensor[f32,8],
    coefficients: tensor[f32,1,5],
    scales: tensor[f32,1],
    state: tensor[f32,1,2])
    -> (tensor[f32,8], tensor[f32,1,2]):
  return sos_tdf2(input, coefficients, scales, state, contract=fast)
