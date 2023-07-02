from model import DQN
from rbuffer import ReplayMemory, Transition
import torch
import torch.optim as optim
import torch.nn.functional as F
import random

class Agent:
    def __init__(self,
                    replay_buffer_size    = 10000,
                    batch_size            = 40,
                    gamma                 = 0.98,
                    eps_start             = 1,
                    eps_end               = 0.12,
                    eps_steps             = 300,
                    learning_rate         = 0.001,
                    input_dim             = 18,
                    hidden_dim            = 120,
                    output_dim            = 2,
                    target_update         = 10
                 ):
        self.replay_buffer_size     = replay_buffer_size
        self.batch_size             = batch_size
        self.gamma                  = gamma
        self.eps_start              = eps_start
        self.eps_end                = eps_end
        self.eps_steps              = eps_steps
        self.learning_rate          = learning_rate
        self.input_dim              = input_dim
        self.hidden_dim             = hidden_dim
        self.output_dim             = output_dim
        self.target_update          = target_update
        self.device                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training               = True

        self.policy_net             = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net             = DQN(input_dim, hidden_dim, output_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer              = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory                 = ReplayMemory(self.replay_buffer_size)
        
        self.steps                  = 0
        self.cumulative_reward      = []

    def select_action(self, state):
        """
        Epsilon greedy action selection
        """
        state       = state.unsqueeze(0).unsqueeze(1)
        state       = state.to(torch.float32)

        sample      = random.random()

        if self.training:
            if self.steps > self.eps_steps:
                eps_threshold   = self.eps_end
            else:
                eps_threshold   = self.eps_start
        else:
            eps_threshold       = self.eps_end

        self.steps  += 1

        if sample > eps_threshold:
            with torch.no_grad():
                evaluated = self.policy_net(state).squeeze()
                # print(evaluated)
                # res = torch.tensor([evaluated.argmax()], device=self.device, dtype=torch.float32)
                # print(res)
                # print("eval: ", evaluated)
                return evaluated 
        else:
            return torch.rand((2,), device=self.device, dtype=torch.float32)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions             = self.memory.sample(self.batch_size)
        # print(transitions)
        
        batch                   = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        nfns                    = [s for s in batch.next_state if s is not None]
        non_final_next_states   = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states   = non_final_next_states.unsqueeze(1)
        non_final_next_states   = non_final_next_states.to(torch.float32)

        state_batch             = torch.cat(batch.state).view(self.batch_size, -1)
        state_batch             = state_batch.unsqueeze(1)
        state_batch             = state_batch.to(torch.float32)
        action_batch            = torch.cat(batch.action).view(self.batch_size, -1)
        reward_batch            = torch.cat(batch.reward).view(self.batch_size, -1)

        state_action_values     = self.policy_net(state_batch)

        # next_state_values       = torch.zeros(self.batch_size, device=self.device)
        # print(next_state_values)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values       = self.target_net(non_final_next_states)
        next_state_values       = next_state_values.view(self.batch_size, -1)
        # print(next_state_values)

        expected_state_action_values    = (next_state_values * self.gamma) + reward_batch

        state_action_values     = state_action_values.squeeze(1)
        loss                    = F.mse_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
