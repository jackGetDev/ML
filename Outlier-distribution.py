
import seaborn as sns
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')
st.write("# Distribution dataset tips - Check Outlier")
st.write("Dataset:https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
df_tips = sns.load_dataset('tips')
df=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
st.write("## EDA")
st.write(df.head())
st.write("## Shape: ",df.shape)
st.write("## Columns: ",df.columns)
st.write("## Regplot ")
sns.regplot(x = "total_bill", y = "tip", data = df_tips)
plt.show()
st.pyplot()

sns.set(style="whitegrid")
ax = sns.boxplot(x=df_tips["total_bill"])

st.write('## See correlation')
numerics_feats=df.dtypes[df.dtypes != 'object'].index

st.write("## Columns - Numerics",numerics_feats)

st.write("## Heatmap")
cormat = df[numerics_feats].corr()
plt.subplots(figsize=(12,9))
sns.heatmap(cormat, vmax=0.9, square=True)
st.pyplot()


st.write("## Finding outlier: Total Bill")
st.write("### Total Bill")
st.write(df_tips["total_bill"].head())
st.write("## Boxplot")
sns.boxplot(df_tips["total_bill"])
st.pyplot()
st.write("## Distplot")
sns.distplot(df_tips["total_bill"])
st.pyplot()

st.write("## Skew the total bill: ",df_tips["total_bill"].skew())
st.write("## Describe the total bill:",df_tips["total_bill"].describe())

st.write("## IQR: Interquartile range")
data_min=df_tips["total_bill"].loc[df_tips["total_bill"].values<=13.347500]
data_max=df_tips["total_bill"].loc[df_tips["total_bill"].values<=24.127500]
frames = [data_min, data_max]
st.write("## Data Fix IQR:")                                
df_fix_first=pd.concat(frames)
st.write(df_fix_first.head())
st.write("## Describe :",df_fix_first.describe())
st.write("## Distplot")
sns.distplot(df_fix_first)
st.pyplot()
st.write("## Skew : ",skew(df_fix_first))
st.write("## Describe :",df_fix_first.describe())
st.write("## BoxPlot")
sns.boxplot(df_fix_first)
st.pyplot()

