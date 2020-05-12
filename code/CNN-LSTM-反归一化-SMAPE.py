import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math

from keras.losses import mean_squared_error, mean_absolute_error
import seaborn as sns
from pylab import rcParams, concatenate
from pandas import concat
from keras.models import load_model
from matplotlib.font_manager import _rebuild
# import matplotlib.pyplot as plt

import matplotlib
from sklearn.metrics import r2_score
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.matplotlib_fname()

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


# basic_model = load_model('basic_model.h5')
complex_model = load_model('complex_history.h5')


# In[3]:


y_predict = complex_model.predict(X_valid_vals)
print(y_predict.shape)
# print(y_predict)
# print(X_valid_vals.shape[0], X_valid_vals.shape[2])
# print(X_valid_vals.shape[2])
X_valid_vals = X_valid_vals.reshape((X_valid_vals.shape[0], X_valid_vals.shape[2]))



# y_predict = basic_model.predict(X_valid_vals)
# # print(X_valid_vals.shape[0], X_valid_vals.shape[2])
# X_valid_vals = X_valid_vals.reshape((X_valid_vals.shape[0], X_valid_vals.shape[2]))
# In[4]:


# print(X_valid_vals)


# In[6]:


# print(X_valid_vals[:, :6])


# In[22]:


inv_y_test = concatenate((X_valid_vals[:, :-18], y_predict), axis=1)
# print(X_valid_vals.shape)
# print(y_predict.shape)


# print(inv_y_test)

inv_y_test = scaler.inverse_transform(inv_y_test)


cnn_err = []
lstm_err = [12.369701521119149, 14.7466508495142, 13.345048456987394, 18.298128590257377, 14.891460474908113, 12.601071116008415, 11.109774705495226, 14.861891217172696, 14.54783480604698, 16.648073477916547, 16.47462024513047, 13.957299724274142, 12.926662368645674, 11.414201601837835, 14.88070214812528, 12.482193277280563, 11.959331849531212, 14.72887690784542, 11.540450632827506, 14.982621860790099, 10.498560184847344, 10.060509044015937, 16.726837651508212, 27.356492281805245, 17.58049780183956, 16.293380637321984, 26.969229245492294, 26.13614755019182, 22.2670475855303, 36.19320524635871, 22.518250959678472, 24.936861823655658, 20.306806076243294, 18.309409679167114, 14.036782498643666, 14.784336016033656, 16.248993771447385, 26.907879801326644, 21.99489350010348, 30.15576620031436, 18.0055590201928, 17.22502376695429, 22.407837481520165, 13.446557310156996, 11.82178768015394, 17.003906865160573, 15.251718743039861, 11.738710516144172, 20.14759452336761, 13.018222119237322, 13.124329414413927, 13.71885733755496, 16.562413974789294, 18.00001374088092, 14.028710765008082, 13.210948831491592, 15.810225499794237, 14.21587373940926, 18.729138210655957, 23.645024502998876, 13.8904929560419, 16.514479048282745, 25.24283668156915, 37.31493194082551, 24.34539287788209, 19.422522570783045, 27.212503083593287, 28.891851722102963, 25.411743597626707, 44.96255841408628, 18.518456431311428, 23.17154962637547, 12.31889995054624, 17.353552221898653, 11.843795067502635, 12.095707515385898, 15.114786108066342, 15.738899782173487, 14.955969962116491, 24.10436556695128, 21.77395544207705, 13.825629529992161, 22.5633070163628, 18.078117996704638, 14.387212585737943, 15.071714747369908, 15.417423010687376, 15.741323968160566, 19.944216061720148, 19.597233855958898, 13.603763060671929]

for item in range(520,611):
    inv_y_predict = inv_y_test[:, item]
    # print(len(inv_y_predict))

    # plt.plot(inv_y,color='red',label='Original')
    # plt.plot(inv_y_predict,color='green',label='Predict')
    # plt.legend()
    # plt.show()

    # print(inv_y_test[:, 1:])

    # print(len(y_valid))

    # print(y_valid.values)
    # inv_y_train = concatenate((X_valid_vals[:,:-18], y_valid.values), axis=1)
    #
    # inv_y_train = scaler.inverse_transform(inv_y_train)
    #
    # print(inv_y_train)
    #
    # inv_y = inv_y_train[:, -1]

    # inv_x_valid = concatenate((X_train_vals[:,:-18], X_valid_vals), axis=1)
    #
    # inv_x_valid = scaler.inverse_transform(inv_x_valid)
    #
    # # print(inv_y_train)
    #
    # inv_x = inv_x_valid[:, -1]

    inv_y_valid = concatenate((X_valid_vals[:, :-37], X_valid.values), axis=1)

    inv_y_valid = scaler.inverse_transform(inv_y_valid)

    inv_y_valid_pre = inv_y_valid[:, item]


    # In[97]:
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(inv_y, marker='o', color='black',label='Original')
    # plt.plot(inv_y_predict, linestyle='--', marker='*', color='black',label='LSTM-Predicted')
    # plt.xlabel('日期')
    # plt.ylabel('销售量')
    # plt.title('2019.1—2019.3 50号商品的销售额预测对比')
    # plt.legend()
    # plt.savefig("test.png", dpi=750, bbox_inches = "tight")
    # plt.show()

    # # In[58]:
    #
    #
    # print(inv_y)
    #
    #
    # # In[67]:
    #
    #
    # print(len(inv_y))
    def smape(y_true, y_pred):
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


    # 回归评价指标
    # calculate MSE 均方误差
    mse = mean_squared_error(inv_y_predict, inv_y_valid_pre)
    # calculate RMSE 均方根误差
    rmse = math.sqrt(mean_squared_error(inv_y_predict, inv_y_valid_pre))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(inv_y_predict, inv_y_valid_pre)
    # calculate R square
    r_square = r2_score(inv_y_predict, inv_y_valid_pre)
    # smape
    smape_val = smape(inv_y_predict, inv_y_valid_pre)

    # print('均方误差: %.6f' % mse)
    # print('均方根误差: %.6f' % rmse)
    # print('平均绝对误差: %.6f' % mae)
    # print('smape: %.6f' % smape_val)
    cnn_err.append(smape_val)
    # print('R_square: %.6f' % r_square)


print(cnn_err)
plt.xlabel('SMAPE')
plt.ylabel('频率')
plt.title('单一模型与混合模型误差分布')
sns.distplot(cnn_err, bins=30,label='CNN-LSTM模型')
sns.distplot(lstm_err, bins=30,label='LSTM模型')
plt.legend()
plt.show()

# TODO:LSTM  CNN-LSTM  Xgb


