
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
    def __init__(self, tickerName_t, credit_t, df_t, idt_t, raised_to_t):

        # Trader Id 
        self.id = idt_t
        # The ticker
        self.ticker             = tickerName_t

        # Finantial and statistical data of the stock.
        self.stockData          = df_t
        
        # Variable for the network
        #self.alive              = True

        # Trader credit
        self.original_credit    = credit_t
        self.credit             = credit_t

        # Current position or holdings of stock
        self.holdings           = 0

        self.bought             = 0

        self.selled             = 0

        self.last_credit        = credit_t

        self.raised_to          =   raised_to_t

        self.credit_hist        = []
        self.holdings_hist      = []
        self.selled_hist        = []
        self.bought_hist        = []

    def alive(self):
        return True if self.credit > 0 else False

    # This method preprocess the data
    def prepareData(self):
        self.closings  = self.stockData["Close"].to_numpy().tolist()
        self.stockData = so.calculateAll(self.stockData)
        self.stockData = so.cleanDataFrame(self.stockData, 10)
        so.normalizeDataset(self.stockData)
        self.input_list = self.stockData.to_numpy().tolist()

    # Method for placing an order
    def buy(self, quantity_t):
        if (quantity_t * (10**self.raised_to)) > 0 and (quantity_t * (10**self.raised_to)) * self.closings[0] < self.credit:
            self.holdings = self.holdings + (quantity_t * (10**self.raised_to))
            self.credit = self.credit - ((quantity_t * (10**self.raised_to)) * self.closings[0])
            self.bought = ((quantity_t * (10**self.raised_to)) * self.closings[0])

            self.bought_hist += [(quantity_t * (10**self.raised_to)) * self.closings[0]]
        else:self.bought_hist += [0]


    # Method for selling stock
    def sell(self, quantity_t):
        #self.selled = -1
        if (quantity_t * (10**self.raised_to)) > 0 and (quantity_t * (10**self.raised_to)) < self.holdings:
            #print(f"=====>>>>>    This is the quantity for selling option : {quantity_t}")
            self.holdings = self.holdings - (quantity_t * (10**self.raised_to))
            self.credit = self.credit + ((quantity_t * (10**self.raised_to)) * self.closings[0])
            self.selled = ((quantity_t * (10**self.raised_to)) * self.closings[0])

            self.selled_hist += [(quantity_t * (10**self.raised_to)) * self.closings[0]]
        else: self.selled_hist += [0]
            


    # Method that calculates the earnings
    def getEarnings(self):
        return self.holdings * self.closings[0] 
    
    # Prints all the current positions.
    def getCurrentStatus(self):
        #
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
        current_credit = self.credit * (10**-self.raised_to)
        current_holdings = self.holdings * (10**-self.raised_to)
        return self.input_list.pop(0) + [current_credit, current_holdings]
    
    def update(self):
        # here i should add the method for doint what is must do
        # takeaction()
        # if self.credit > self.original_credit:
        self.last_credit = self.credit
        self.credit_hist += [self.credit]
        self.holdings_hist += [self.holdings]
        if self.credit != self.original_credit:
            #print(f"id : {self.id}\t\t\tcredit : {self.credit}\t\tholdings: {self.holdings}\t\tearnings: {self.getEarnings()}\t\tbought: {self.bought}\t\tselled: {self.selled}")
            pass
            #self.bought = 0
            #self.selled = 0
