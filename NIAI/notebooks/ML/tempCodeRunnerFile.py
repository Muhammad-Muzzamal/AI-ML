def func1(L1, L2):
    ans = []
    ptr1 = ptr2 = 0

    while ptr1 < len(L1) and ptr2 < len(L2):
        if L1[ptr1] <= L2[ptr2]:
            ans.append(L1[ptr1])
            ptr1 += 1
        else:
            ans.append(L2[ptr2])
            ptr2 += 1
    
    while ptr1 < len(L1):
        ans.append(L1[ptr1])
        ptr1 += 1
    
    while ptr2 < len(L2):
        ans.append(L1[ptr2])
        ptr1 += 1
    
    return ans


print(func1([1, 2, 3], [1, 2, 3]))