"""
题目描述
    矩阵A规模为n×m，矩阵B规模为m×p，現要你求A×B.
    矩阵相乘的定义:n×m的矩阵与m×p的矩阵相乘变成n×p的矩阵，
    令a¡k为矩阵A中的元素，b¡k为矩阵B中的元素，则相乘所得矩阵C中的元素
    C¡j=∑a¡kc×b¡k(1≤k≤m)
    具体可见样例。
输入
    第一行两个数n,m;
    接下来n行m列描述一个矩阵A;
    接下来一行输入p;
    接下来m行p列描述一个矩阵B。
输出
    输出矩阵A与矩阵 B相乘所得的矩阵C。
"""
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
        '''
        C[i].append(Temp)
for i in range(n):
    for j in range(p):
        print(C[i][j], end=' ' if j != p - 1 else '\n')
