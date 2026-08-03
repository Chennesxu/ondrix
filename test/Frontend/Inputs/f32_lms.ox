def f32_lms(
    input: tensor[f32,32], desired: tensor[f32,32],
    weights: tensor[f32,4]) -> (tensor[f32,32], tensor[f32,4]):
  return lms(input, desired, weights, step_size=[1, 16], contract=fma)
