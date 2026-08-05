def f32_sos_tdf2(
    input: tensor[f32],
    coefficients: tensor[f32,2,5],
    scales: tensor[f32,2],
    state: tensor[f32,2,2])
    -> (tensor[f32], tensor[f32,2,2]):
  return sos_tdf2(input, coefficients, scales, state, contract=off)
