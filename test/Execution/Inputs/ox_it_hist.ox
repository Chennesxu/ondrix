def it_hist(x: buffer[i32], h: mut buffer[i64]) -> i32:
  missed = 0
  for i in range(len(x)):
    b = x[i]
    if b >= 0 and convert(b, to=index) < len(h):
      h[b] = h[b] + 1
    else:
      missed = missed + 1
  return missed
