# -*- coding: utf-8 -*-
__author__ = 'Mike'
import pandas as pd
from collections import OrderedDict
import ast

# 示例数据
data = {'ID': [1, 2, 3],
        'OrderedDictArray': [
            OrderedDict([('age', 25), ('city', 'New York')]),
            OrderedDict([('age', 30), ('city', 'San Francisco')]),
            OrderedDict([('age', 22), ('city', 'Los Angeles')])
        ]}

df = pd.DataFrame(data)

# 存储 DataFrame 到 CSV 文件
df.to_csv('your_file.csv', index=False)

# 读取 CSV 文件时指定 Ordered Dict 列的转换函数
def parse_ordered_dict(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string

# 读取 CSV 文件，并在读取时应用转换函数
df_read = pd.read_csv('your_file.csv', converters={'OrderedDictArray': parse_ordered_dict})

# 打印读取后的 DataFrame
print(df_read)
