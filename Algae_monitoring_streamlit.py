import time, base64
import streamlit as st
import pyalgae_ai.analysis as AiAnal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from hydroeval import evaluator, nse, kge
from PIL import Image

im = Image.open("AI_Lab_logo.jpg")

st.set_page_config(
    page_title="Sentinel-2 위성영상과 AI분석을 통한 녹조 분석 Web App",
    page_icon=im,
    layout="wide",
)

col1, col2 = st.columns([1,8])
col1.image(im, width=100)
col2.write("# Sentinel-2 위성영상과 AI분석을 통한 녹조 분석")

st.write("##### (STEP-1) 분석을 위한 데이터 파일 가져오기")
col1, col2 = st.columns([1,2])
csv_file = col1.file_uploader("Select Your Local Observation CSV file")
if csv_file is not None:
    df = pd.read_csv(csv_file)
else:
    use_example = st.checkbox('Use Example File', False)
    if use_example: 
        df = pd.read_csv('./data/20220224_sentinels_and_algae_obs.csv1')
    else:
        st.stop()
st.write("###### **ㅇ 입력자료현황 (위성, 녹조)**")
st.write(df.head())

st.write("##### (STEP-2) 데이터를 위성자료와 녹조자료로 분리하기")

col1, col2, col3 = st.columns([2,2,8])
divider1 = col1.number_input(label="녹조관측자료의 첫번째 열",
                        min_value=0, max_value=len(list(df)), value=1)
divider2 = col2.number_input(label="위성관측자료의 첫번째 열",
                        min_value=1, max_value=len(list(df)), value=6)

input = df.columns[divider2:len(list(df))]
label = df.columns[divider1:divider2]
input_sentinel = df[input]
label_algae = df[label]
chk_seperate = st.checkbox('학습자료 분리')

col1, col2 = st.columns([1.5,1])
if chk_seperate:
    with col1:
        st.write("###### **ㅇ 위성영상자료**")
        st.write(input_sentinel.head())
    with col2:
        st.write("###### **ㅇ 녹조관측자료**")
        st.write(label_algae.head())
else:
    st.stop()

col1, col2 = st.columns([1,2])
col1.write("##### (STEP-3) 분광특성밴드 조합 선택")
col2.write("##### (STEP-4) 모델, Training 크기 선택")

with st.form('My Form'):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    select_columns1 = col1.selectbox('위성영상의 밴드조합', options=[['B1 B2 B3'],
                                     ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12 AT CLOUD']])
    select_columns2 = col2.selectbox('위성영상의 밴드조합', options=[['B1 B2'],
                                     ['B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B11 B12']])

    select_columns = [select_columns1, select_columns2]

    model = col3.selectbox('AI적용 모델선택', options=[["RF"], ["GBR"], ["XGB"]])
    trainSize_rate = col4.number_input(
            label="Training 데이터의 비율", min_value=0.0, max_value=1.0, step=.1, value=0.8)
    n_estimators = col5.number_input(
            label="분석할 가지의 갯수 지정", min_value=0, max_value=2000, step=100, value=200)
    select_metrics = col6.selectbox('성능평가방법 선택', options=["NSE", "MSE", "MAE", "RMSE", "R2", "KGE"])

    submit = st.form_submit_button('모델분석')
    #submit = st.button('모델분석 시행')

    if submit:
        with st.spinner('Wait for it...'):
            results = AiAnal.algae_monitor(input_sentinel, label_algae, select_columns, model, trainSize_rate, n_estimators, random_state=42)
            total_result = []
            for i in range(len(results)):
                score_train, score_test = AiAnal.performance_test(select_metrics, results[i])
                f_result = '"{}" and "{}"의 결과: score_train={}, score_test={}'.format(' '.join(list(results[i][1])), results[i][2].name, score_train, score_test)
                total_result.append(f_result)
            time.sleep(100)
            st.success("성공적으로 위성영상을 활용한 녹조 분석이 시행되었습니다.")
            st.write(total_result)