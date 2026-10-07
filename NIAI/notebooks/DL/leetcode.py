
string = "abcdabc"

def func(string):
  map = []
  for i in range(len(string)):
    if string[i] not in map:
      map.append(string[i])
    else:
      return i

print(func(string))