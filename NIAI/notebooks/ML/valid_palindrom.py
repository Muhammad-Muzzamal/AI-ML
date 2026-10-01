def func(L):
  start = 0
  end = len(L) - 1

  while(start < end):
    if (L[start] == L[end]):
      start += 1
      end -= 1
    else:
      return False
      

  return True

print(func([1, 2, 3, 1]))