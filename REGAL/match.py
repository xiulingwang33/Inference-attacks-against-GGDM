import sys
import math

Maxn = 505
inf = float('inf')


def match(u):
    x, y, yy, delta = 0, 0, 0, 0
    pre = [0] * (Maxn)
    slack = [inf] * (Maxn)
    visx = [False] * (Maxn)
    visy = [False] * (Maxn)
    matched[y] = u
    while True:
        x = matched[y]
        delta = inf
        visy[y] = True
        for i in range(1, n + 1):
            if visy[i]:
                continue
            if slack[i] > ex[x] + ey[i] - map[x][i]:
                slack[i] = ex[x] + ey[i] - map[x][i]
                pre[i] = y
            if slack[i] < delta:
                delta = slack[i]
                yy = i
        for i in range(n + 1):
            if visy[i]:
                ex[matched[i]] -= delta
                ey[i] += delta
            else:
                slack[i] -= delta
        y = yy
        if matched[y] == -1:
            break
    while y:
        matched[y] = matched[pre[y]]
        y = pre[y]


def KM():
    global matched, ex, ey
    matched = [-1] * (Maxn)
    ex = [0] * (Maxn)
    ey = [0] * (Maxn)
    for i in range(1, n + 1):
        visy = [False] * (Maxn)
        match(i)
    res = 0
    for i in range(1, n + 1):
        if matched[i] != -1:
            res += map[matched[i]][i]
    return res


if __name__ == "__main__":
    n = int(input())
    map = [[-inf] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            map[i][j] = int(input())

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            map[i][j] = -map[i][j]
    print(-KM())
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            map[i][j] = -map[i][j]
    print(KM())
