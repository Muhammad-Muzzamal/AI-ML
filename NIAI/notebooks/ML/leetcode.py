input = "[({})]"

def func(input):
    stack = []

    pairs = {
        ')': '(',
        '}': '{',
        ']': '['
    }

    for i in input:

        if (i == '(' or i == '{' or i == '['):
            stack.append(i)
        else:
            # if not stack:
            #     return False

            if stack[-1] == pairs[i]:
                stack.pop()
            else:
                return False

    return len(stack) == 0


print(func(input))