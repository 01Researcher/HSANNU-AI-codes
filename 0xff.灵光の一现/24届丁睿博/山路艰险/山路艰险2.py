"""
题目描述
    蒜头君看着眼前从左到右在一条线上的n座山峰，
    他想从中选出两座山峰，作为旅行的起点和终点，要求选出的较左边的山峰高度大于选出的较右边的山峰，
    定义这样选择后这次旅行的困难程度为两座山峰的高度差。
    问所有可能的选择方案困难程度之和是多少。
输入格式
    输入有两行:
        第一行为一个整数n，表示山峰的数目(1≤n≤10^5)
        第二行为几个空格隔开的整教a¡,为每座山峰的高度(1≤a¡≤10^9)
输出格式
    输出一行，包含一个整数，表示答案。
"""
n, li, s = int(input()), input().split(' '), 0
for i in range(n):
    li[i] = int(li[i])

for i in range(1, n):
    for j in range(n - i):
        '''
        # 严格判定每个左侧的山都高于右侧, 并非题目要求, 仅作参考
        temp = 0
        for k in range(1, i + 1):
            if li[j+k-1] > li[j + k]:
                temp += li[j] - li[j + k]
            else:
                temp = 0
                break
        s += temp
        '''
        # 仅判定起点山高于终点山
        temp = li[j] - li[j + i]
        s += temp if temp > 0 else 0

print(s)
