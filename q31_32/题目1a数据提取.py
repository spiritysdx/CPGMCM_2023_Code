# import pandas as pd
# data1=pd.read_excel('D:/DESKTOP/表1.xlsx')[['ID','数据集划分','入院首次影像检查流水号','发病到首次影像检查时间间隔']]
# data2=pd.read_excel('D:/DESKTOP/表2.xlsx')[['ID','首次检查流水号','HM_volume','随访1流水号','HM_volume','随访2流水号','HM_volume','随访3流水号','HM_volume','随访4流水号','HM_volume','随访5流水号','HM_volume','随访6流水号','HM_volume','随访7流水号','HM_volume','随访8流水号','HM_volume']]
# 
# data1a=pd.merge(data1,data2,how='outer',on='ID')
# data1a.to_excel('D:/DESKTOP/1a数据.xlsx',index=False)

import pandas as pd
import numpy as np
datat=np.array(pd.read_excel('D:/DESKTOP/附表1.xlsx'))[:,2:20]
set={}
for i in range(len(datat)):
    for j in range(len(datat[i])):
        if j%2==0:
            set[datat[i][j+1]]=datat[i][j]
data=pd.read_excel('D:/DESKTOP/1a数据.xlsx')[['ID','入院首次影像检查流水号','随访1流水号','随访2流水号','随访3流水号','随访4流水号','随访5流水号','随访6流水号','随访7流水号','随访8流水号']]
for i in range(len(list(data.columns))):
    if data.columns[i]!='ID':
        for j in range(len(data[data.columns[i]])):
            if data[data.columns[i]][j]>1:
                data[data.columns[i]][j]=set[data[data.columns[i]][j]]
data.to_excel('D:/DESKTOP/时间点.xlsx',index=False)
        