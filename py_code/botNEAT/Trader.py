
"""
Dataframe
- Open
- High
- Low
- Adj Close
- Volume
- Returns
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average(WMA)
- Average True Range (ATR)
- (ATR_WMA)
- (ATR_EMA)
- Momentum
- ROC

Additional Dataframe cols (More like constants)
- Current earnings / loosings
- Current holdings
- Trailing P/E ratio
- Forward P/E ratio
- PEG ratio
"""
import sys
sys.path.append("..")
import stocksOps as so

class Trader():
    def __init__(self, tickerName_t, credit_t):

        # The ticker
        self.ticker             = tickerName_t

        # Finantial and statistical data of the stock.
        self.stockData          = so.downloadIntradayData(tickerName_t)
        
        # Variable for the network
        self.alive              = True

        # Trader credit
        self.original_credit    = credit_t
        self.credit             = credit_t

        # Current position or holdings of stock
        self.holdings           = 0


    # Method for placing an order 
    def buy(self, quantity_t):
        pass

    # Method for selling stock
    def sell(self, quantity_t):
        pass
    
    # Method that calculates the earnings
    def getEarnings():
        pass
    
    # Prints all the current positions.
    def getCurrentStatus():
        pass

    # Returns the input for the network as an array
    # of len of the number of defined inputs
    def data(self):
        """
        input = [Open, High, Low, Adj Close, Volume, 
                 Returns, SMA, EMA, WMA, ATR, ATR_WMA,
                 ATR_EMA, Momentum, ROC,

                 CurrentEarnings,
                 Current holdings, Trailing P/E ratio,
                 Forward PE ratio, PEG ratio]
        """
        # TODO define the inputs on the config file
        pass
    
    def update(self):
        pass