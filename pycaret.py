import pandas as pd
from pycaret.datasets import get_data
import yfinance as yf
from pycaret.time_series import *
import plotly.express as px

def addfreq(data):
  data['Date']=data['Date'].astype(str)
  data['Date'] = pd.to_datetime(data['Date'])
  data = data.groupby('Date').sum()
  data = data.asfreq(freq ='D')
  data['Close'] = data['Close'].ffill()
  return data

def prediction_model_generate(data):
  setup(data, fh = 7, fold = 3, session_id = 123)
  best = compare_models()
  return finalize_model(best)

def predictor(model, days):
  prediction = predict_model(model, fh = days)
  return prediction.to_frame()

def complete_data(data, prediction):
  complete = data.append(prediction)
  complete.reset_index(level=0, inplace=True)
  complete.columns = ['Date','Close']
  complete = addfreq(complete)
  complete['MA50']=complete['Close'].rolling(50).mean()
  complete['MA200']=complete['Close'].rolling(200).mean()
  return complete

def predict(ticker, days):
  yfin = yf.Ticker(ticker)
  data = yfin.history(period="max")
  data = data.dropna()
  data = data[['Close']]
  data.reset_index(level=0, inplace=True)
  data['Date'] = pd.to_datetime(data['Date'])
  data = addfreq(data)
  model = prediction_model_generate(data)
  prediction = predictor(model, days)
  print(prediction)
  complete = complete_data(data, prediction)
  fig = px.line(complete)
  fig.show()
  
predict("ETH-USD",90)