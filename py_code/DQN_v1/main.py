from environment import Environment
from agent import Agent

if __name__ == "__main__":
    ticker          = "NVDA"
    original_credit = 100
    num_episodes    = 500

    agent           = Agent()
    env             = Environment(ticker, original_credit)
    
    cumulative_reward   = [0 for t in range(num_episodes)]

    for i_episode in range(num_episodes):

        env.reset()

        steps_counter   = 0
        # definition of first state
        state       = env.get_state()

        while not env.done and env.credit>0 and env.holdings >= 0:
            action          = agent.select_action(state)
            reward, done    = env.step_(action)
            cumulative_reward[i_episode] += reward.item()

            if env.credit > env.original_credit:
                print(f"episode: {i_episode}, step: {steps_counter},credit: {env.credit}, holdings: {env.holdings}")

            try:
                next_state      = env.get_state()
            except:
                break
            # print(next_state)

            agent.memory.push(state, action, next_state, reward)

            state           = next_state
            
            agent.optimize_model()

            steps_counter += 1

        if i_episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if env.credit > env.original_credit and 0 == 1:
            print(f"credit: {env.credit}, holdings: {env.holdings}")
            print(f"episode: {i_episode}, steps: {steps_counter}")
            print("=================")


