def it_pick(t: tensor[i64, 8], i: i32) -> i64:
  return t[i] - t[7 - i]
