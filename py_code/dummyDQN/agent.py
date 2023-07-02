import torch
import torch.optim as optim
import torch.nn as nn
import random
import math
from model import DQN
from rbuffer import ReplayBuffer, Transition

class Agent:
    def __init__(self, batch_size_t, gamma_t, eps_start_t, eps_end_t, eps_decay_t, tau_t, lr_t,
                 input_layer_dims_t, hidden_layer_dims_t, output_layer_dims_t):
        # Definition of basic attributed
        self.batch_size     = batch_size_t
        self.gamma          = gamma_t
        self.eps_start      = eps_start_t
        self.eps_end        = eps_end_t
        self.eps_decay      = eps_decay_t
        self.tau            = tau_t
        self.lr             = lr_t

        #  Network models definition
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net     = DQN(input_layer_dims_t, hidden_layer_dims_t, output_layer_dims_t).to(self.device)
        self.target_net     = DQN(input_layer_dims_t, hidden_layer_dims_t, output_layer_dims_t).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Define optimizer
        self.optimizer      = optim.AdamW(self.policy_net.parameters(), lr=lr_t, amsgrad=True)
        self.memory         = ReplayBuffer(10000)        # TODO add this to parameters init

        self.steps_done     = 0

    def choose_action(self, state_t):
        sample              = random.random()
        eps_threshold       = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

        self.steps_done += 1

        if sample > eps_threshold: 
            with torch.no_grad():
                action = self.policy_net(state_t)
                # print("raw action: ", action)
                # action = action.max(1)
                # print("max action: ", action)
                # action = action[1]
                # action = action.view(1, 1)
                # print("view 1, 1 action: ", action)
                # action = self.policy_net(state_t).max(1)[1].view(1, 1)
        else:
            # action = torch.randint(low=0, high=1, size=(1,), device=self.device, dtype=torch.long)
            action = torch.rand((1, 2), device=self.device, dtype=torch.float64)

        return action
    
    # This is just where ill set the step condition
    def step(self, action_t, credit_t, holdings_t, stock_price_t, original_credit_t, raise_to_t):
        """
        return alive, reward, credit_t, holdings_t
        """
        # buy
        buy_action = action_t[0][0].item()
        credit_t -= (buy_action*(raise_to_t))
        if credit_t <= 0:
            return False, 0, credit_t, holdings_t
        holdings_t += (buy_action*(raise_to_t)) / stock_price_t 

        # sell
        sell_action = action_t[0][1].item()
        credit_t += (sell_action*(raise_to_t))
        holdings_t -= (sell_action*(raise_to_t)) / stock_price_t 
        if holdings_t <= 0:
            return False, 0, credit_t, holdings_t

        # reward
        reward = 0
        if credit_t > original_credit_t:
            reward += credit_t / original_credit_t
        total = credit_t+ (holdings_t*stock_price_t)
        if total > original_credit_t:
            reward += (holdings_t*stock_price_t) / total
        return True, reward, credit_t, holdings_t 



    def optimizer_func(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask  = torch.tensor(tuple(map(lambda s: s is not None,
                                                 batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch     = torch.cat(batch.state).requires_grad_(True)
        action_batch    = torch.cat(batch.action).unsqueeze(1)
        # reward_batch    = torch.cat(batch.reward)
        reward_batch    = torch.zeros((self.batch_size, 2), device=self.device).requires_grad_(True)

        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = self.policy_net(state_batch)
        state_action_values.requires_grad_()
        # state_action_values = state_action_values.gather(1, action_batch)
        # state_action_values = state_action_values.gather(0, action_batch.view(-1, 1).long())
        
        next_state_values   = torch.zeros((self.batch_size, 2), device=self.device)
        with torch.no_grad(): 
            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            target_res = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = target_res
            print(next_state_values.shape)
            print(reward_batch)
            print(reward_batch.shape)
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            expected_state_action_values.requires_grad = True

            # Compute Huber loss 
            criterion   = nn.SmoothL1Loss()
            # Check this line, the dims of both action
            print("same shape?: ", state_action_values.shape, " : ", expected_state_action_values.shape)
            # loss        = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            loss        = criterion(state_action_values, expected_state_action_values)


            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            optimizer.step()
