import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np 
import plotly.express as px

from keras.models import load_model
import matplotlib.pyplot as plt
# streamlit run modified.py


st.title('STOCK TREND PREDICTOR')
ticker = st.sidebar.text_input('Ticker','SBIN.NS')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

data = yf.download(ticker, start=start_date,end=end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)
st.plotly_chart(fig)


pricing_data, fundamental_data, news , Stock_Market_Predictor= st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News","Stock Market Predictor"])

with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace = True)
    st.write(data)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual Return is ',annual_return,'%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standaard Deviation is ',stdev*100,'%')
    st.write('Rick Adj, Return is',annual_return/(stdev*100))


from alpha_vantage.fundamentaldata import FundamentalData
with fundamental_data:
    key = '8P60BPVRM72EAV9S'
    fd = FundamentalData(key,output_format = 'pandas')
    st.subheader('Balance Sheet')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    st.subheader('Income Statement')
    income_statement = fd.get_balance_sheet_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    st.subheader('Cash Flow Statement')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)


from stocknews import StockNews
with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker , save_news=False)
    df_news = sn.read_rss()
    for i in range(10):
        st.header(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment  {i+1}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
        

with Stock_Market_Predictor:
    model = load_model('Stock Predictions Model.keras')

    st.header('Stock Market Predictor')

    stock =st.text_input('Enter Stock Symnbol','SBIN.NS')
    start = '2014-01-24'
    end = '2024-01-24'

    data = yf.download(stock, start ,end)

    st.subheader('Stock Data')
    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)
