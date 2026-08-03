def invalid_f32_gain_magnitude(input: tensor[f32,16]) -> tensor[f32,16]:
  return gain(input, gain=[548055821, 548055723], contract=off)
