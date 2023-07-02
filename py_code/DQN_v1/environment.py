import torch
import numpy as np

import sys
sys.path.append("..")
import stocksOps as so

class Environment:
    def __init__(self, ticker_name_t, original_credit_t):
        # Init dataframe data
        self.df                 = so.downloadIntradayData(ticker_name_t)
        # self.closings           = torch.from_numpy(df["Close"].to_numpy())
        self.closings           = self.df["Close"].to_numpy().tolist()
        self.df                 = so.calculateAll(self.df)
        self.df                 = so.cleanDataFrame(self.df, 10) 
        so.normalizeDataset(self.df)
        self.input_list         = self.df.to_numpy()
    
        self.raise_to_exp       = 4
        self.raise_to_pos       = 10**(self.raise_to_exp)
        self.raise_to_neg       = 10**(-self.raise_to_exp)

        self.original_credit    = original_credit_t
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.reward_f   = reward if reward == "sr" else "profit"
        self.reset()

    def reset(self):
        self.step       = 0
        self.done       = False
        self.credit     = self.original_credit
        self.record     = self.original_credit
        self.holdings   = 0
        self.lastCredit = self.credit

    def get_state(self):
        if not self.done:
            state       = self.input_list[self.step].copy()
            to_append   = [self.original_credit*self.raise_to_neg,
                            self.credit*self.raise_to_neg,
                            self.holdings*self.raise_to_neg
                            ]
            state       = np.append(state, to_append)
            state       = torch.from_numpy(state)
            return state
        else:
            None

    def buy(self, action):
        temp            = self.credit
        temp            -= (action*(self.raise_to_pos))
        if temp <= 0:
            return -1
        self.credit     = temp
        self.holdings   += (action*(self.raise_to_pos)) / self.closings[self.step]
        return 1
        
    def sell(self, action):
        temp = self.holdings
        temp -= (action*(self.raise_to_pos)) / self.closings[self.step]
        if temp< 0:
            return -1
        self.holdings = temp
        self.credit     += (action*(self.raise_to_pos))
        return 1

    def step_(self, action):
        
        self.lastCredit = self.credit
        
        reward = 0
        reward += self.buy(action[0].item())
        reward += self.sell(action[1].item())

        if self.credit > self.record:
            self.record = self.credit
            reward += 10
         
        if self.credit > self.original_credit and self.holdings >= 0:
            # reward += self.credit / self.original_credit
            reward += 2
       
        total = self.credit + (self.holdings * self.closings[self.step])
        if total > self.original_credit and self.holdings > 0 and self.credit >= 0:
            reward += (self.holdings*self.closings[self.step]) / total
            reward += 1
        if self.credit > self.lastCredit:
            reward += self.credit / self.original_credit
            # reward += 3

        self.step += 1
        # print(self.step)
        if self.step == len(self.closings):
            self.done   = True
        return torch.tensor([reward], device=self.device, dtype=torch.float32), self.done
