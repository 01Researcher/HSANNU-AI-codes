a, b = input(), input()
n, s = -max(len(a), len(b)) - 1, ''
for i in range(-1, n, -1):
    s = str((int(a[i]) + int(b[i])) % 10) + s
print(s)

# 好神经的题目
