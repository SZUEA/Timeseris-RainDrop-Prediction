
import pandas as pd
import numpy as np
# # 读取CSV文件
# df = pd.read_csv('/public/lay/eaai/data/data_10.csv')
# # soc = np.random.rand()
# # 合并Year、Mon、Day和Hour列，并创建一个名为'date'的新列
# df['date'] = df['Year'].astype(str) + '-' + df['Mon'].astype(str) + '-' + df['Day'].astype(str) + '-' + df['Hour'].astype(str)
# df['PRE_1h']  = np.random.rand(df.shape[0])
# df = df[['Station_Name','date','Alti','TEM','DPT','RHU','WIN_D_INST','WIN_S_INST','GST','PRE_1h']]
# df.to_csv('/public/lay/eaai/data/test_data.csv', index=False)

number = 744-96
label_num = 721
a = np.load('/public/lay/eaai/Informer2020/results/informer_ETTh1_ftMS_sl96_ll0_pl5_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1/pred.npy')
for data in a:
    print(a)
# print(len(a))
# print(a.shape)
# pred = pd.read_excel('/public/lay/eaai/data/forcast_1.xlsx', dtype=float)
# for i in range(len(pred)):
#     if i % label_num >= 96:
#         class_num = i//label_num
#         label_num_ = class_num*number + (i % number)-96
#         pred.loc[i]['实际1h降雨量'] = a[label_num_][0]
#         pred.loc[i]['实际3h降雨量'] = a[label_num_][2]
#         pred.loc[i]['实际6h降雨量'] = a[label_num_][4]
#         # print(pred.loc[i])
#
# pred.to_excel('output_.xlsx', index=False)
