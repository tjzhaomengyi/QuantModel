# -*- coding: utf-8 -*-
__author__ = 'Mike'
import pandas as pd

# 示例 DataFrame，其中的 'DictColumn' 列包含多个元素的字典
data = {'ID': [1, 2, 3],
        'DictColumn': [{'age': 25, 'city': 'New York', 'gender': 'female'},
                       {'age': 30, 'city': 'San Francisco', 'gender': 'male'},
                       {'age': 22, 'city': 'Los Angeles', 'gender': 'female'}]}

df = pd.DataFrame(data)

# 使用 apply 结合 pd.Series 拆解 'DictColumn' 列
df = df['DictColumn'].apply(pd.Series).merge(df, left_index=True, right_index=True)

# 删除原始 'DictColumn' 列
df = df.drop('DictColumn', axis=1)

print(df)
