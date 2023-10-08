#Here we get the necessary datas from the SpaceApp website,Remove first columns(xyz vectors and Datetime).
#We also have a file called "output.csv" which is the Tp index measurements since 1937
#We use this data to train and validate our Anomaly Detection Model


import pandas as pd
df=pd.read_csv('output.csv')
df=df.iloc[:,:8]
values_list = df.values.ravel().tolist()
values_list=values_list[:2823]
existing_df=pd.read_csv('dsc_fc_summed_spectra_2016_v01.csv')
existing_df1=pd.read_csv(r'C:\Users\ilhan\Downloads\dsc_fc_summed_spectra_2017_v01\dsc_fc_summed_spectra_2017_v01.csv')
existing_df2=pd.read_csv(r'C:\Users\ilhan\Downloads\dsc_fc_summed_spectra_2018_v01\dsc_fc_summed_spectra_2018_v01.csv')
existing_df3=pd.read_csv(r'C:\Users\ilhan\Downloads\dsc_fc_summed_spectra_2019_v01\dsc_fc_summed_spectra_2019_v01.csv')
existing_df4=pd.read_csv(r'C:\Users\ilhan\Downloads\dsc_fc_summed_spectra_2023_v01\dsc_fc_summed_spectra_2023_v01.csv')

existing_df5=existing_df1.iloc[:,4:]
existing_df=existing_df.iloc[:,4:]
existing_df4=existing_df4.iloc[:,4:]
existing_df3=existing_df3.iloc[:,4:]
existing_df2=existing_df2.iloc[:,4:]



result_df = existing_df2.round(3)
result_df.to_csv('result1.csv')



