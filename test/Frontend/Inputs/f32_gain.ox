def f32_gain(input: tensor[f32,16]) -> tensor[f32,16]:
  return gain(input, gain=[3, 8], contract=off)
