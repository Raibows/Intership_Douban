

# 题：您需要编写一个程序，按升序对（年龄，高度）进行排序，age和height是数字。（使用pandas）

# 排序标准是：

# 1：根据年龄排序;

# 2：然后按分数排序。

# 优先级是name> age>得分。

# 如果给出以下元组作为程序的输入：

# Tom,19,80

# John,20,90

# Jony,17,91

# Jony,17,93

# Json,21,85

import pandas as pd

data = [
    ('Tom',19,80),
    ('John',20,90),
    ('Jony',17,91),
    ('Jony',17,93),
    ('Json',21,85)
]

df = pd.DataFrame(data, columns=['name', 'age', 'height'])
print(df.sort_values(by=['name', 'age', 'height']))