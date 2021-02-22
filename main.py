import streamlit as st
import pandas_datareader as pdr

st.write('''
# 테스트입니다.
''')

df = pdr.get_data_yahoo('035420.KS', '2020-01-01', '2020-12-01')
df = df.dropna()

st.line_chart(df.Close)
st.line_chart(df.Volume)