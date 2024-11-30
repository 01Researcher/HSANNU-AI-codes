n = int(input())
li = input().split(' ')
s = 0
for i in range(n):
    li[i] = int(li[i])
# print(li)
for i in range(n-1):
    temp = li[i] - li[i+1]
    s = s if s >= temp else temp
print(s)
