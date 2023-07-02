import torch
from agent import Agent
import sys
sys.path.append("..")
import stocksOps as so

if __name__ == "__main__":
    # Definition of contant model variables
    batch_size      = 128
    gamma           = 0.99
    eps_start       = 0.9
    eps_end         = 0.05
    eps_decay       = 1000
    tau             = 0.005
    lr              = 1e-4

    # Definition of network model dims
    input_layer_dims    = 18
    hidden_layer_dims   = 128 
    output_layer_dims   = 2

    agent = Agent(batch_size, gamma, eps_start, eps_end, eps_decay, tau, lr, input_layer_dims,
                  hidden_layer_dims, output_layer_dims)

    # Prepare data
    df                  = so.downloadIntradayData("NVDA")
    closings            = df["Close"].to_numpy().tolist()
    df                  = so.calculateAll(df)
    df                  = so.cleanDataFrame(df, 10)
    so.normalizeDataset(df)
    input_list          = df.to_numpy().tolist()

    # trader data
    raise_to            = (10**4)
    raise_to_neg        = (10**-4)

    num_episodes        = 180

    for i_episode in range(num_episodes):
        # Prepare Trader data
        original_credit     = 100
        credit              = original_credit
        holdings            = 0
        alive               = True

        counter_step = 0
        state = input_list[counter_step] 
        state = state.copy()
        state += [original_credit*raise_to_neg, credit*raise_to_neg, holdings*raise_to_neg]
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = state      # dummy init
        while alive and state != None:
            action  = agent.choose_action(state)
            print(action)
            alive, reward, credit, holdings = agent.step(action, credit, holdings, closings[counter_step], original_credit, raise_to)
            reward  = torch.tensor([reward])

            next_state = None
            if counter_step + 1< len(input_list):
                next_state = input_list[counter_step+1]
                next_state = next_state.copy()
                next_state += [original_credit*raise_to_neg, credit*raise_to_neg, holdings*raise_to_neg]
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)

            state = next_state

            # Optimize model
            agent.optimizer_func()

            target_net_state_dict   = agent.target_net.state_dict() 
            policy_net_state_dict   = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
            agent.target_net.load_state_dict(target_net_state_dict) 




            counter_step += 1
    print("Done")
