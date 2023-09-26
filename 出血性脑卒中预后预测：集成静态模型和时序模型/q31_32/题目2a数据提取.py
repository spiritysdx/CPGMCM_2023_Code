import pandas as pd
columns=[[f'随访{i+1}流水号',f'ED_volume.{i+1}'] for i in range(8)]
columns=['ID','首次检查流水号','ED_volume.0']+[i for j in columns for i in j]
data=pd.read_excel('D:/DESKTOP/表2.xlsx')[columns]

data.to_excel('D:/DESKTOP/2a数据.xlsx')