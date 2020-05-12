import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import describe
import seaborn as sns
from pylab import rcParams, concatenate
from pandas import concat
from keras.models import load_model
from matplotlib.font_manager import _rebuild

from matplotlib.pyplot import MultipleLocator

_rebuild()

pd.options.display.max_columns = 12
pd.options.display.max_rows = 24

warnings.simplefilter('ignore')

# get_ipython().run_line_magic('matplotlib', 'inline')

sns.set(style='darkgrid', palette='muted')
color_scheme = {
    'red': '#F1637A',
    'green': '#6ABB3E',
    'blue': '#3D8DEA',
    'black': '#000000'
}

# 所有绘图使用svg格式
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

# 增加图片大小
rcParams['figure.figsize'] = 8, 6

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# 处理数据，最终要让数据的格式都是500个销售列和每天一行，如果不这么做，怎么预测到每个店，每个商品的销售呢？

df_train.head()

df_train.index = pd.to_datetime(df_train['date'])
df_train.drop('date', axis=1, inplace=True)
df_test.index = pd.to_datetime(df_test['date'])
df_test.drop('date', axis=1, inplace=True)

from itertools import product, starmap


def storeitems():
    return product(range(1, 51), range(1, 11))


def storeitems_column_names():
    return list(starmap(lambda i, s: f'item_{i}_store_{s}_sales', storeitems()))


def sales_by_storeitem(df):
    ret = pd.DataFrame(index=df.index.unique())
    for i, s in storeitems():
        ret[f'item_{i}_store_{s}_sales'] = df[(df['item'] == i) & (df['store'] == s)]['sales'].values
    return ret

df_train = sales_by_storeitem(df_train)
# 对于测试集，我们只是用0填充y值
df_test['sales'] = np.zeros(df_test.shape[0])
df_test = sales_by_storeitem(df_test)

# 数据组合起来为模型做准备，然后将其分解为训练集和测试集。

# 确保所有列名都是相同的，并且顺序相同
col_names = list(zip(df_test.columns, df_train.columns))
for cn in col_names:
    assert cn[0] == cn[1]

df_test['is_test'] = np.repeat(True, df_test.shape[0])
df_train['is_test'] = np.repeat(False, df_train.shape[0])
df_total = pd.concat([df_train, df_test])
df_total.info()

df_total.head()

# ### 特征工程
# 使用One-hot编码日，一周和一个月，以确保网络识别数据的季节性。

weekday_df = pd.get_dummies(df_total.index.weekday, prefix='weekday')
weekday_df.index = df_total.index

month_df = pd.get_dummies(df_total.index.month, prefix='month')
month_df.index = df_total.index

df_total = pd.concat([weekday_df, month_df, df_total], axis=1)

assert df_total.isna().any().any() == False


# 如果是单步预测我们还希望将前一天的sales追加到每一行，然后将其用作输入数据。
def shift_series(series, days):
    return series.transform(lambda x: x.shift(days))


def shift_series_in_df(df, series_names=[], days_delta=90):
    ret = pd.DataFrame(index=df.index.copy())
    str_sgn = 'future' if np.sign(days_delta) < 0 else 'past'
    for sn in series_names:
        ret[f'{sn}_{str_sgn}_{np.abs(days_delta)}'] = shift_series(df[sn], days_delta)
    return ret


def stack_shifted_sales(df, days_deltas=[1, 90, 360]):
    names = storeitems_column_names()
    dfs = [df.copy()]
    for delta in days_deltas:
        shifted = shift_series_in_df(df, series_names=names, days_delta=delta)
        dfs.append(shifted)
    return pd.concat(dfs, axis=1, sort=False, copy=False)

df_total = stack_shifted_sales(df_total, days_deltas=[1])
df_total.dropna(inplace=True)


# 我们需要确保堆叠和非堆叠后的数据的销售列以相同的顺序排列。通过对名称(作为字符串)进行排序来实现这一点

sales_cols = [col for col in df_total.columns if '_sales' in col and '_sales_' not in col]
stacked_sales_cols = [col for col in df_total.columns if '_sales_' in col]
other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

sales_cols = sorted(sales_cols)
stacked_sales_cols = sorted(stacked_sales_cols)

new_cols = other_cols + stacked_sales_cols + sales_cols

df_total = df_total.reindex(columns=new_cols)

df_total.head()

assert df_total.isna().any().any() == False

# ### 数据的归一化

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler(feature_range=(0, 1))
cols_to_scale = [col for col in df_total.columns if 'weekday' not in col and 'month' not in col]
scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
df_total[cols_to_scale] = scaled_cols

# ### 设置一下训练集和测试集，这个测试集是真正的测试集

df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)
df_test = df_total[df_total['is_test'] == True].drop('is_test', axis=1)

# ## 训练模型
# 首先，我们需要将训练数据分离为输入向量和目标向量，并将部分训练数据分离为模型的验证数据。
X_cols_stacked = [col for col in df_train.columns if '_past_' in col]
X_cols_caldata = [col for col in df_train.columns if 'weekday_' in col or 'month_' in col or 'year' in col]
X_cols = X_cols_stacked + X_cols_caldata

X = df_train[X_cols]

X_colset = set(X_cols)
y_cols = [col for col in df_train.columns if col not in X_colset]

y = df_train[y_cols]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)

# 使用2019年第一季度进行验证
X_valid, y_valid = X_valid.head(90), y_valid.head(90)

# 对于Keras，需要对输入值进行进一步的转换，才能够输入到模型中
X_train_vals = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid_vals = X_valid.values.reshape((X_valid.shape[0], 1, X_valid.shape[1]))


# In[2]:


basic_model = load_model('basic_model.h5')
complex_model = load_model('complex_history.h5')


# In[3]:


y_predict = basic_model.predict(X_valid_vals)
# print(X_valid_vals.shape[0], X_valid_vals.shape[2])
y_cplex_predict =  complex_model.predict(X_valid_vals)

X_valid_vals = X_valid_vals.reshape((X_valid_vals.shape[0], X_valid_vals.shape[2]))



inv_y_test = concatenate((X_valid_vals[:,:-18], y_predict), axis=1)

inv_y_cplex_test = concatenate((X_valid_vals[:,:-18], y_cplex_predict), axis=1)


inv_y_test = scaler.inverse_transform(inv_y_test)
inv_y_cplex_test = scaler.inverse_transform(inv_y_cplex_test)

# 周期
day = 90
# LSTM-Predicted
inv_y_predict = inv_y_test[:day, -1]
# CNN-LSTM-Predicted
print(inv_y_predict)
inv_y_cplex_predict = inv_y_cplex_test[:day, -1]

inv_y_train = concatenate((X_valid_vals[:,:-18], y_valid.values), axis=1)

inv_y_cplex_train = concatenate((X_valid_vals[:,:-18], y_valid.values), axis=1)

inv_y_train = scaler.inverse_transform(inv_y_train)

# ORIGINAL
inv_y = inv_y_train[:day, -1]


# In[97]:
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# x_major_locator=MultipleLocator(10)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator=MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
# ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)

scale_ls = range(day)
index_ls = [x for x in range(5, day+1, 5)]
# index_ls.append(day)
plt.xticks(index_ls)
plt.plot(inv_y, marker='o', color='black',label='原始数据')
plt.plot(inv_y_predict, linestyle='--', marker='*', color='black',label='离线预测')
plt.plot(inv_y_cplex_predict, linestyle=':', marker='^', color='black',label='在线预测')
plt.xlabel('天数')
plt.ylabel('销售量')
plt.title('2020.1.1—2020.3.30 10号店铺的50号商品的90天销售额预测对比')
plt.legend()
plt.savefig("compareed-90days.png", dpi=750, bbox_inches = "tight")
plt.show()

# 




