import pandas as pd
from matplotlib import pyplot

# read data
data = pd.read_csv('./data/环境卫生课题/上海市.csv')

# plot data
for column in data.columns:
    if column != 'date':
        data[column].plot()
        pyplot.show()

# interpolate
interpolated = data.interpolated(method='akima')

# save
interpolated.to_csv('./data/interpolated_data.csv')

# 使用后两年的同日数据模拟当日o3数据
