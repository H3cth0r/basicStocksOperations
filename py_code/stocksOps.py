import yfinance		as yf
import pandas 		as pd
import numpy 		as np
import math
from sklearn.preprocessing import StandardScaler

"""
===========================================================================
This is a collection of methods for calculating some stocks operations.
===========================================================================
"""


def downloadData(tickerName_t, startDate_t, endDate_t):
	"""
	@brief 	This method is intended to download data from yahoo finance
		
	@params	tickerName_t	:	This is the ticker that must be downloaded.
		
		startDate_t	:	This is the start date. Format \"2021-09-19\".

		endDate_t	:	This is the end date. Format \"2021-09-19\".
	
	@return	pandas dataframe created by yahoo library
	"""
	return yf.download(tickerName_t, start=startDate_t, end=endDate_t)

def downloadIntradayData(tickerName_t):
	"""
	@brief	This method downloads intraday data, on a period
			of 7 days, each minute
	@params	tickerName	: This is the ticker to download.

	@return	pandas datafram created by yahoo library
	"""
	return yf.download(tickers=tickerName_t, period="7d", interval="1m", ignore_tz = False, group_by = 'ticker')

def saveDataFranceToDevice(dataFrameObject_t, saveFileName_t):
	"""
	@brief	This method downloads the data.

	@params	dataFrameObject	:	yahoo lib pandas dataframe.

	@return	Returns True if the download was succesfull.
	"""
	try:
		dataFrameObject_t.to_csv(saveFileName_t)
		return True
	except:
		print(f"Error while trying to store the file \"{saveDataFranceToDevice}\" as csv.")
		return False

def downloadTickerStatistics(tickerName_t):
	res = yf.Ticker(tickerName_t)
	return res


"""
===========================================================================
Functional Methods / Operations.
===========================================================================
"""

def closingReturns(dataFrame_t):
	"""
	@brief	This method is intended to calculate the closing returns
			column out of the \"Adj Close column\". It will add
			new column to the data frame an return the new dataFrame

	@params	dataFrame_t	the dataframe of the downloaded data

	@return	This method returns the same dataframe, \"returns\"
	"""
	returnsPerRow	=	[]
	adjsc			=	dataFrame_t['Adj Close'].to_numpy()
	for i in range(len(adjsc)):
		if i < len(adjsc) - 1:
			returnsPerRow.append(adjsc[i] / adjsc[i+1]-1)
		else:
			returnsPerRow.append(0)
	dataFrame_t["returns"]	=	returnsPerRow

def averageReturns(dataFrame_t):
	"""
	@brief	This method is intended to get the number that
			describes the mean of the total closing returns.
	
	@params	dataFrame	the dataframe of the stocks data.

	@return	returns a number that represents the mean or
			expected value of the returns column.
	"""
	return dataFrame_t["returns"].mean()

def varianceReturns(dataFrame_t, averageReturns_t):
	"""
	@brief	Method that calculates the total variance returns

	@params	dataFrame_t			the datafram of the stocks data
			averageReturns_t	number that represents the calculation
								of the average returns.
	"""
	returns = dataFrame_t["returns"].to_numpy()
	diff_sqd = []
	for i in range(len(returns)):
		diff_sqd.append((returns[i] - averageReturns_t)**2)
	averageDiffSqd = np.sum(diff_sqd) / len(diff_sqd)
	return averageDiffSqd

def stdDeviation(varianceReturns_t):
	"""
	@brief	method that calculates the standard deviation by using the
			result of the varianceReturns method.
	
	@params	varianceReturns_t	number that represents the total variance returns
								calculated.
	
	returns 	the standard deviation.
	"""
	return varianceReturns_t**(1/2)

def covariance(xDataFrame, yDataFrame, xAverage, yAverage):
	"""
	@brief	method for calculating the covariance. for this, we take 
			data from two datasets / dataframes of the selected
			stockss.

	@params	xDataFrame		the first dataframe / dataset
			yDataFrame		the second dataframe / dataset
			xAverage		first dataset average returns
			yAverage		secodn dataset average returns
	
	@returns	returns a douvle number that will indicate
				the covariance between a pair of stocks.
	"""
	x = xDataFrame["returns"].to_numpy()
	y = yDataFrame["returns"].to_numpy()
	xy = []
	for i in range(len(x)):
		xy.append(x[i]*y[i])
	covar = (np.sum(xy)/len(xy)) - (xAverage * yAverage)
	return covar

def correlation(xyCovar_t, xVar, yVar):
	"""
	@brief	method that calculates the correlation between to stocks

	@params	xyCovar_t	The calculated covariance between two stocks.
			xVar		stock X average returns
			yVar		stock Y average returns
	
	@returns	returns a number that represent the correlation
				between a pair of stocks
	"""
	return xyCovar_t / (math.sqrt(xVar) * math.sqrt(yVar))

def compoundInterest(P, r, t):
	"""
	@brief	Method for calculating the compound interest of an stock

	@params P		Initial investment(amount of money)
			r		Annual rate of return
			t		number of years to invest
	
	@returns	returns a number that represents the compund interest.
	"""
	return P * (1 + r)**t

def moneyMadeInYear(P, r, t):
	"""
	@brief	Method for calculating the ammount of moeny your
			money is making in given year. It applies the compound
			interest method.

	@params P		Initial investment(amount of money)
			r		Annual rate of return
			t		number of years to invest
	
	@returns	returns a number that represents the compund interest.
	"""
	return compoundInterest(P, r, t) * r

def compoundInterestTime(r):
	"""
	@brief	Method for calculating when the ammount of money you make
			each year will be the same as the annount of money put in.
	
	@params	r	Annual rate of return

	@returns the number of years taken
	"""
	return -np.log(r)/np.log(1 + r)

def expectedValue(avgLoss, avgLP, avgGain, avgGP):
	"""
	@brief	method that calculates the number that depending on the average loss and
			average gain per trade, will tell us if in the long run, mathematically,
			the current strategy will garanty to be fine or not.
	
	@params	avgLoss		average loss, amount o loss per trade and probability
						of loss.
			avgLP
			avgGain
			avgGP
	
	@returns	will return a number thata represente the expected value
	"""
	return ((avgLoss) * avgLP) + (avgGain*avgGP)

def calculateSMA(df_t, numberOfdays_t):
	"""
	@brief	Calculates the Simple Moving Average. Average price over
			specified period. 
	
	@params	df_t				This is the dataframe of the stocks
			numberOfDays_t		Number of days or period to calculate the
								simple moving average.
	
	@returns	The method retursn an array of the SMA, calculated
				over the specified period.
	"""
	SMA = []
	for i in range(0, numberOfdays_t-1):
		SMA.append(0)
	for i in range(numberOfdays_t, len(df_t.Close)+1):
		sum = 0
		for j in range(i-numberOfdays_t, i):
			sum += df_t.Close[j]
		SMA.append(sum/numberOfdays_t)
	return SMA

def calculateEMA(df_t, numberOfDays_t):
	"""
	@brief	Calculates the Exponential Moving Average.

	@params	df_t				dataFrame with the stocks data
			numberOfDays_t		period over wich is calculated the moving
	
	@returns	returns array of the EMA
	"""
	multiplier = (2/(numberOfDays_t+1))
	EMA = []
	first = True
	for i in range(0, numberOfDays_t-1):
		EMA.append(0)
	for i in range(numberOfDays_t-1, len(df_t.Close)):
		if first:
			res = df_t.Close[i] * multiplier + df_t.SMA[i] * (1-multiplier)
			first = False
		else:
			res = df_t.Close[i] * multiplier + EMA[-1] * (1-multiplier)
		EMA.append(res)
	return EMA

def calculateWMA(df_t, numberOfDays_t):
	"""
	@brief	Calculates the weighted moving average

	@params	df_t			This dataframe with the stocks data
			numberOfDays	the period of days to calculate
							the average
	
	@returns	a list containing the calculated WMA
	"""
	WMA = []
	weight = 0
	for i in range(1, numberOfDays_t+1):
		weight += i
	for i in range(0, numberOfDays_t-1):
		WMA.append(0)
	for i in range(numberOfDays_t, len(df_t.Close) + 1):
		sum = 0
		counter = 1
		for j in range(i-numberOfDays_t, i):
			sum += df_t.Close[j] * (counter/weight) 
			counter += 1
		WMA.append(sum)
	
	return WMA

def calculateATR(df_t, numberOfDays_t):
	"""
	@brief	Calculates the Average True Range

	@params	df_t			dataframe with stocks data
			numberOfDays_t	period in which traverse the column to calculate
							the average true range
	
	@returns	a list with the calcualted values of the true range
	"""
	trueRange = []
	ATR = []

	for i in range(0, len(df_t.Close)):
		if i > 0:
			trueRange.append(max([abs(df_t.High[i] - df_t.Low[i]), abs(df_t.High[i] - df_t.Close[i-1]), abs(df_t.Low[i] - df_t.Close[i-1])]))
		else:
			trueRange.append(max([abs(df_t.High[i] - df_t.Low[i]), 0, 0]))

	for i in range(numberOfDays_t+1):
		ATR.append(0)
	for i in range(numberOfDays_t+1, len(df_t.Close)):
		sum = 0
		for j in range(i-numberOfDays_t, i):
			#print(j)
			sum += trueRange[j]
		ATR.append(sum/numberOfDays_t)
	return ATR



def calculateATRwma(df, numberOfDays_t):
	"""
	@brief	Calculates the average true range 
			applying the weighted moving average

	@params	df_t			the dataframe with the stocks data
			numberOfDays	period in which traveres the column

	@returns 				a list with the resulting data
	"""
	a = 2 / (numberOfDays_t + 1)
	trueRange = []
	for i in range(0, len(df.Close)):
		if i > 0:
			trueRange.append(max([abs(df.High[i] - df.Low[i]), abs(df.High[i] - df.Close[i-1]), abs(df.Low[i] - df.Close[i-1])]))
		else:
			trueRange.append(max([abs(df.High[i] - df.Low[i]), 0, 0]))
	
	# Calculating weight
	weight = 0
	for i in range(1, numberOfDays_t+1):
		weight += i

	ATR_WMA = []
	for i in range(0, numberOfDays_t-1):
		ATR_WMA.append(0)
	for i in range(numberOfDays_t, len(trueRange)+1):
		sum = 0
		counter = 1
		for j in range(i-numberOfDays_t, i):
			sum += trueRange[j] * (counter/weight)
			counter += 1
		ATR_WMA.append(sum)
	return ATR_WMA

def calculateATRema(df, numberOfDays_t):
	"""
	@brief	Calculates the average true range applying the
			exponential moving average.
	
	@params	df_t			the dataframe with the stocks data
			numberOfDays_t	period to calculate the atr
	
	@returns	a list the calculated atr
	"""
	a = 2 / (numberOfDays_t + 1)
	trueRange = []
	for i in range(0, len(df.Close)):
		if i > 0:
			trueRange.append(max([abs(df.High[i] - df.Low[i]), abs(df.High[i] - df.Close[i-1]), abs(df.Low[i] - df.Close[i-1])]))
		else:
			trueRange.append(max([abs(df.High[i] - df.Low[i]), 0, 0]))
	ATR = []
	for i in range(len(trueRange)):
		if i == 0:
			ATR.append(a*trueRange[i]+(1-a)*trueRange[i])
		else:
			ATR.append(a*trueRange[i]+(1-a)*ATR[i-1])
	return ATR

def calculateMomentum(df_t, period_t):
	"""
	@brief	Calculates the momentum

	@params	df_t		the dataframe with the stocks data
			period_t	period in which the momentum is calculated
	
	@returns	list with the calculated momentum over specified period
	"""
	momentum = []
	for i in range(period_t-1):
		momentum.append(0)
	for i in range(period_t-1, len(df_t.Close)):
		momentum.append(df_t.Close[i] - df_t.Close[i-period_t])
	return momentum

def calculateROC(df_t, period_t):
	"""
	@brief	Calculate the rate of change

	@params	df_t		the dataframe with the stocks data
			period_t	period in which the momentum is calculated
	
	@returns	returns a list with the calcula
	"""
	ROC = []
	for i in range(period_t-1):
		ROC.append(0)
	for i in range(period_t-1, len(df_t.Close)):
		ROC.append(((df_t.Close[i] - df_t.Close[i-period_t])/df_t.Close[i-period_t])*100)
	return ROC

# ====================== ROE
def calculateROE():
	"""
	@brief	Calculates the Return On Equity
	"""
	pass

def standardDeviationMethod(arr):
	"""
	@brief	Calculates the standard deviation over an array

	@params	arr	the array over which the standard deviation is calculated

	@returns	Returns the calculate standard deviation
	"""
	sum = 0
	for i in arr:
		sum += i
	mean = sum / len(arr)
	sum = 0
	for i in range(len(arr)):
		sum += abs(arr[i] - mean) ** 2
	return math.sqrt(sum/len(arr)), mean

def calculateBollinger(df_t, period_t):
	MA = []
	MSD = []
	b1 = []
	b2 = []
	for i in range(period_t):
		MA.append(0)
		MSD.append(0)
		b1.append(0)
		b2.append(0)
	MA = calculateSMA(df_t, 20)

	for i in range(period_t, len(df_t.Close)):
		std, mean = standardDeviationMethod(df_t.Close[i-period_t:i])
		b1.append(MA[i] + std * (mean/std))
		b2.append(MA[i] - std * (mean/std))
	print(b1)
	print(b2)


# ====================== MACD
def calculateMACD():
	pass

# ====================== RSI
def calculateRSI():
	pass

def calculateAll(df_t):
	"""
	@brief	Method that makes the calculations of most of the
			Indicators developed
	
	@params	The dataframe with the stock data

	@returns	the modified dataframe
	"""
	closingReturns(df_t)
	df_t["SMA"] = calculateSMA(df_t, 7)
	df_t["EMA"] = calculateEMA(df_t, 7)
	df_t["WMA"] = calculateWMA(df_t, 7)
	df_t["ATR"] = calculateATR(df_t, 7)
	df_t["ATR_WMA"] = calculateATRwma(df_t, 7)
	df_t["ATR_EMA"] = calculateATRema(df_t, 7)
	df_t["Momentum"] = calculateMomentum(df_t, 10)
	df_t["ROC"] = calculateROC(df_t, 10)
	return df_t

def normalizeDataset(df_t):
	normalizer = StandardScaler()
	df_t[["Open", "High", "Low", "Close", "Adj Close", "Volume", "returns", "SMA", "EMA",
		  "WMA", "ATR", "ATR_WMA", "ATR_EMA", "Momentum", "ROC"]] = normalizer.fit_transform(df_t[["Open", "High", "Low", "Close", "Adj Close", "Volume", "returns", "SMA", "EMA",
		  "WMA", "ATR", "ATR_WMA", "ATR_EMA", "Momentum", "ROC"]])

def anotherNormalizerDataset(df_t):
	normalizer = StandardScaler()
	df_t[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = normalizer.fit_transform(df_t[["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
	
