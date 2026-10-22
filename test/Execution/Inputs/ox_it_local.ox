def it_local(x: buffer[i32], y: mut buffer[i64]):
  t = zeros(i64, 4)
  w = x[1:]
  for i in range(len(w)):
    k = i - i / 4 * 4
    t[k] = t[k] + convert(w[i], to=i64) * 65536
  for j in range(len(y)):
    y[j] = t[j]
