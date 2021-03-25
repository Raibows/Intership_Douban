'''
如何基于给定的 1 维数组创建 strides？

问题：给定 1 维数组 arr，使用 strides 生成一个 2 维矩阵，其中 window length 等于 4，strides 等于 2，例如
[[0,1,2,3], [2,3,4,5], [4,5,6,7]..]。

输入：
arr = np.arange(15) arr#> array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

期望输出：#
> [[ 0 1 2 3]#> [ 2 3 4 5]#> [ 4 5 6 7]#> [ 6 7 8 9]#> [ 8 9 10 11]#> [10 11 12 13]]
'''

import numpy as np

UPPER = 15
arr = np.arange(UPPER)
window_len = 4
strides = 2

# 0 + (n-1) * strides + window_len - 1 <= 14
# n <= (15 - window_len) / strides + 1

n = (arr.size - window_len) // strides + 1

res = np.array([arr[a:a+window_len] for a in range(0, n*strides, strides)])

print(res)