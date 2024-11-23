# 矩阵乘法
li = input().split(' ')
n, m = int(li[0]), int(li[-1])


def Inputer(x):
    li_temp = list()
    for flag in range(x):
        li_temp.append(input().split(' '))
    return li_temp


A, p, B, C = Inputer(n), int(input()), Inputer(m), list()
for i in range(n):
    C.append(list())
    for j in range(p):
        Temp = 0
        for k in range(m):
            Temp += int(A[i][k]) * int(B[k][j])
            '''
            # 调试
            # print(f'A[{i}][{k}] = {A[i][k]} ; B[{k}][{j}] = {B[k][j]}')
        # print(f'C[{i}][{j}] =', Temp, '\n' + '*' * 25)
        # 调试结束
        '''
        C[i].append(Temp)
for i in range(n):
    for j in range(p):
        print(C[i][j], end=' ' if j != p - 1 else '\n')
