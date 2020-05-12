#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy.stats import describe
import seaborn as sns
from pylab import rcParams

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


# In[2]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# 处理数据，最终要让数据的格式都是500个销售列和每天一行，如果不这么做，怎么预测到每个店，每个商品的销售呢？

# In[3]:


df_train.head()


# In[4]:


df_train.index = pd.to_datetime(df_train['date'])
df_train.drop('date', axis=1, inplace=True)
df_test.index = pd.to_datetime(df_test['date'])
df_test.drop('date', axis=1, inplace=True)


# In[5]:


df_train.head()


# In[6]:


from itertools import product, starmap


def storeitems():
    return product(range(1,51), range(1,11))


def storeitems_column_names():
    return list(starmap(lambda i,s: f'item_{i}_store_{s}_sales', storeitems()))


def sales_by_storeitem(df):
    ret = pd.DataFrame(index=df.index.unique())
    for i, s in storeitems():
        ret[f'item_{i}_store_{s}_sales'] = df[(df['item'] == i) & (df['store'] == s)]['sales'].values
    return ret


# In[7]:


df_train = sales_by_storeitem(df_train)
# 对于测试集，我们只是用0填充y值
df_test['sales'] = np.zeros(df_test.shape[0])
df_test = sales_by_storeitem(df_test)


# In[8]:


df_train.info()


# In[9]:


df_test.info()


# 数据组合起来为模型做准备，然后将其分解为训练集和测试集。

# In[10]:


# 确保所有列名都是相同的，并且顺序相同
col_names = list(zip(df_test.columns, df_train.columns))
for cn in col_names:
    assert cn[0] == cn[1]


# In[11]:


df_test['is_test'] = np.repeat(True, df_test.shape[0])
df_train['is_test'] = np.repeat(False, df_train.shape[0])
df_total = pd.concat([df_train, df_test])
df_total.info()


# In[12]:


df_total.head()


# ### 特征工程
# 使用One-hot编码日，一周和一个月，以确保网络识别数据的季节性。

# In[13]:


weekday_df = pd.get_dummies(df_total.index.weekday, prefix='weekday')
weekday_df.index = df_total.index


# In[14]:


month_df = pd.get_dummies(df_total.index.month, prefix='month')
month_df.index =  df_total.index


# In[15]:


df_total = pd.concat([weekday_df, month_df, df_total], axis=1)


# In[16]:


assert df_total.isna().any().any() == False


# 如果是单步预测我们还希望将前一天的sales追加到每一行，然后将其用作输入数据。

# In[17]:


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


# In[18]:


df_total = stack_shifted_sales(df_total, days_deltas=[1])
df_total.dropna(inplace=True)


# In[19]:


df_total.info()


# In[20]:


df_total.head()


# 我们需要确保堆叠和非堆叠后的数据的销售列以相同的顺序排列。通过对名称(作为字符串)进行排序来实现这一点

# In[21]:


sales_cols = [col for col in df_total.columns if '_sales' in col and '_sales_' not in col]
stacked_sales_cols = [col for col in df_total.columns if '_sales_' in col]
other_cols = [col for col in df_total.columns if col not in set(sales_cols) and col not in set(stacked_sales_cols)]

sales_cols = sorted(sales_cols)
stacked_sales_cols = sorted(stacked_sales_cols)

new_cols = other_cols + stacked_sales_cols + sales_cols


# In[22]:


df_total = df_total.reindex(columns=new_cols)


# In[23]:


df_total.head()


# In[24]:


assert df_total.isna().any().any() == False


# ### 数据的归一化

# In[25]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[26]:


scaler = MinMaxScaler(feature_range=(0,1))
cols_to_scale = [col for col in df_total.columns if 'weekday' not in col and 'month' not in col]
scaled_cols = scaler.fit_transform(df_total[cols_to_scale])
df_total[cols_to_scale] = scaled_cols


# ### 设置一下训练集和测试集，这个测试集是真正的测试集

# In[27]:


df_train = df_total[df_total['is_test'] == False].drop('is_test', axis=1)
df_test = df_total[df_total['is_test'] == True].drop('is_test', axis=1)


# In[28]:


df_train.head()


# In[29]:


df_test.head()


# ## 训练模型
# 首先，我们需要将训练数据分离为输入向量和目标向量，并将部分训练数据分离为模型的验证数据。

# In[30]:


X_cols_stacked = [col for col in df_train.columns if '_past_' in col]
X_cols_caldata = [col for col in df_train.columns if 'weekday_' in col or 'month_' in col or 'year' in col]
X_cols = X_cols_stacked + X_cols_caldata

X = df_train[X_cols]


# In[31]:


X_colset = set(X_cols)
y_cols = [col for col in df_train.columns if col not in X_colset]

y = df_train[y_cols]


# In[32]:


X.head()


# In[33]:


y.head()


# In[34]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=False)


# 使用2019年第一季度进行验证

# In[35]:


X_valid, y_valid = X_valid.head(90), y_valid.head(90)


# 对于Keras，需要对输入值进行进一步的转换，才能够输入到模型中

# In[36]:


X_train_vals = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid_vals = X_valid.values.reshape((X_valid.shape[0], 1, X_valid.shape[1]))


# ### Defining models

# In[37]:


from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Conv1D, Input, Dropout, AvgPool1D, Reshape, Concatenate


# 单个LSTM

# In[ ]:


basic_model = Sequential()
basic_model.add(LSTM(500, input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])))
basic_model.add(Dense(500))
basic_model.compile(loss='mean_absolute_error', optimizer='adam')


# CNN-LSTM

# In[ ]:


inputs = Input(shape=(X_train_vals.shape[1], X_train_vals.shape[2]))
# 顶部架构
top_lstm = LSTM(500)(inputs)
top_dense = Dense(500, activation='relu')(top_lstm)
top_dropout = Dropout(500)(top_dense)
# 底部架构
bottom_dense = Dense(500)(inputs)
bottom_conv1 = Conv1D(
    500, 
    kernel_size=1,
    input_shape=(X_train_vals.shape[1], X_train_vals.shape[2])
)(bottom_dense)
bottom_conv2 = Conv1D(
    1000,
    kernel_size=50,
    padding='same',
    activation='relu'
)(bottom_conv1)
bottom_conv3 = Conv1D(
    500,
    kernel_size=10,
    padding='same',
    activation='relu'
)(bottom_conv2)
bottom_pooling = AvgPool1D(
    pool_size=60, 
    padding='same'
)(bottom_conv3)
bottom_reshape = Reshape(
    target_shape=[500]
)(bottom_conv3)
# 连接两个架构的输出
final_concat = Concatenate()([top_dropout, bottom_reshape])
final_dense = Dense(500)(final_concat)
# 返回模型
complex_model = Model(inputs=inputs, outputs=final_dense)
complex_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mape'])


# In[ ]:


final_dense


# 现在拟合模型

# In[ ]:


basic_history = basic_model.fit(
    X_train_vals, 
    y_train.values, 
    epochs=60, 
    batch_size=30,
    validation_data=(X_valid_vals, y_valid.values),
    verbose=2,
    shuffle=False
)


# In[ ]:


def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


# In[ ]:


plot_history(basic_history)


# In[ ]:


complex_history = complex_model.fit(
    X_train_vals, 
    y_train.values, 
    epochs=130, 
    batch_size=70,
    validation_data=(X_valid_vals, y_valid.values),
    verbose=2,
    shuffle=False
)


# In[ ]:


# plot_history(complex_history)


# In[ ]:




