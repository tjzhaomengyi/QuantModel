# -*- coding: utf-8 -*-
__author__ = 'Mike'
import pandas as pd
import json
from collections import OrderedDict
import ast
import collections

# 示例数据
# data = {'ID': [1, 2, 3],
#         'OrderedDictArrayString': ['[OrderedDict([("age", 25), ("city", "New York")]), OrderedDict([("age", 30), ("city", "San Francisco")])]',
#                                    '[OrderedDict([("age", 22), ("city", "Los Angeles")]), OrderedDict([("age", 28), ("city", "Chicago")])]',
#                                    '[OrderedDict([("age", 33), ("city", "Miami")])]']}
# OrderedDictArrayString=['[OrderedDict([("age", 25), ("city", "New York")]), OrderedDict([("age", 30), ("city", "San Francisco")])]',
#                                    '[OrderedDict([("age", 22), ("city", "Los Angeles")]), OrderedDict([("age", 28), ("city", "Chicago")])]',
#                                    '[OrderedDict([("age", 33), ("city", "Miami")])]']
# arr = ast.literal_eval(OrderedDictArrayString)
# print(arr)

# df = pd.DataFrame(data)
#
# # 替换字符串中的 OrderedDict 为 placeholder
# df['OrderedDictArrayString'] = df['OrderedDictArrayString'].str.replace('OrderedDict', '__ordered_dict_placeholder__')
#
# # 使用 json.loads 将字符串列转换为 Ordered Dict 数组列
# df['OrderedDictArray2'] = df['OrderedDictArrayString'].apply(lambda x: json.loads(x))
#
# # 替换 placeholder 为 OrderedDict
# df['OrderedDictArray2'] = df['OrderedDictArray2'].apply(lambda x: eval(str(x).replace('__ordered_dict_placeholder__', 'OrderedDict')))
#
# print(df)
#


# 示例字符串
ordered_dict_array_string = '[OrderedDict([("age", 25), ("city", "New York")]), OrderedDict([("age", 30), ("city", "San Francisco")]), OrderedDict([("age", 22), ("city", "Los Angeles")]), OrderedDict([("age", 28), ("city", "Chicago")]), OrderedDict([("age", 33), ("city", "Miami")])]'

# 替换字符串中的 OrderedDict 为 collections.OrderedDict
ordered_dict_array_string = ordered_dict_array_string.replace('OrderedDict', 'collections.OrderedDict')

# 使用 eval 将字符串转换为对象
ordered_dict_array = eval(ordered_dict_array_string)

# 打印对象
for item in ordered_dict_array:
    print(item)




