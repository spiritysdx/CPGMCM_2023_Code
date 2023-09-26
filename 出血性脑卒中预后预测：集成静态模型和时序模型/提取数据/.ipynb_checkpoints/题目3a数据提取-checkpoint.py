import pandas as pd
data1=pd.read_excel('D:/DESKTOP/3a数据.xlsx')
data3_ED=pd.read_excel('D:/DESKTOP/表3.xlsx',sheet_name='ED')
data3_Hemo=pd.read_excel('D:/DESKTOP/表3.xlsx',sheet_name='Hemo')

data3_ED=data3_ED[list(data3_ED.columns)[1:]]
data3_Hemo=data3_Hemo[list(data3_Hemo.columns)[1:]]
data3_ED.columns=[f'{i}.ED' for i in list(data3_ED.columns)]
data3_Hemo.columns=[f'{i}.Hemo' for i in list(data3_Hemo.columns)]
data1=data1.rename(columns={'入院首次影像检查流水号':'流水号'})
data3_ED=data3_ED.rename(columns={'流水号.ED':'流水号'})
data3_Hemo=data3_Hemo.rename(columns={'流水号.Hemo':'流水号'})

data=pd.merge(data1,data3_ED,how='inner',on='流水号')
data=pd.merge(data,data3_Hemo,how='inner',on='流水号')

data.to_excel('D:/DESKTOP/3a数据aaa.xlsx',index=False)