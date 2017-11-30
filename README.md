# data602-assignment3

#### Goal:
This asssignment is to analyze time-series data for equities in the portfolio, building off of assignment #2. The app relies on heavy use of analytics, visual and display charts. The real-time data is used.

#### Data:
The historical data is obtained from Nasdaq website: http://www.nasdaq.com/g00/quotes/historical-quotes.aspx

#### Aims:
**1.Technical indicators**

The following technical indicators together with the historical data are shown in the table after clicking 'Studies' and 'Prediction' on the trade page.
* Global mean (a flat line)
* 5 day moving average
* 20 day moving average
* 5 day moving standard deviation
* 20 day moving standard deviation
* Bollinger bands 2 standard deviation
* Daily returns %
* Daily return mean %
* Price range for the day (bars)
* Daily price difference (high - low)
* ForceIndex
* William %R
* Relative Strength Index(RSI)
* Rate Of Change(ROC), this is daily return too
* Momentum (MOM)
* Average True Range(ATR)
* Moving average convergence divergence (MACD)

Distribution of prices of a stock (Gaussian distribution, aka bell curve) is shown in the figure after clicking 'Studies' on the trade page.

**2.Correlation**

An additional time-series data source is used to correlate with stock prices. A dataset of weather in New York City obtained from WeatherUnderground( https://www.wunderground.com/history/) is used to explore the correlation between any equity in the portfolio and weather. 

The correlation (-1 to 1) is displayed as an a column on the P/L table

**3.Machine Learning**

**Support Vector Machine (SVM)** algorithm is employed to build the model to predict stock price for any of the equities in the portfolio. The accuracy socre is ~0.8. I saw a trend of decrease for several equities tested, which seems not true. "Predicted price" for the future 26 days could be found in the chart and table for the past one year historical data and technical indicators under the 'Prediction' on trade page. 

SVM are based on the idea of finding a hyperplane that best divides a dataset into two classes. The hyperplane as a line that linearly separates and classifies a set of data. Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. The reason why I chose SVM is that SVM is a supervised machine learning algorithm that can be employed for both classification and regression purposes. Dataset is splited into train and test sets (0.8:0.2) and the training set is used to train the model. Then I tried the linear, polumonial, and RBF kernel to test he accurary and the linear kernel showed the best accuracy in prediction. 

I like SVM because the accuracy score of SVM is quite high(0.8) and the computation is fast . It can be more efficient because it uses a subset of training points. But SVM is more commonly used in classification problems. It may not good at predicting the future stock price.



