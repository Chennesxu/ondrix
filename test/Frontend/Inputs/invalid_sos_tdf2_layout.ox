def invalid_sos_tdf2_layout(
    input: tensor[f32,8],
    coefficients: tensor[f32,1,4],
    scales: tensor[f32,1],
    state: tensor[f32,1,2])
    -> (tensor[f32,8], tensor[f32,1,2]):
  return sos_tdf2(input, coefficients, scales, state, contract=off)
