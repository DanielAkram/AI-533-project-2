import numpy as np
import matplotlib.pyplot as plt
import random

def build_mdp(gamma=0.95, theta=1e-6):
    mdp = {}
    mdp["grid_size"] = (4, 4)
    states = []
    for x in range(4):
        for y in range(4):
            if (x, y) == (3, 3):
                states.append((x, y, 0, 0))  # goal state
            elif (x, y) in [(0, 1), (0, 2)]:
                states.append((x, y, 0, 1))  # fire states
            elif (x, y) in [(2, 1), (2, 2)]:
                states.append((x, y, 1, 0))  # water states
            else:
                states.append((x, y, 0, 0))  # normal states
    
    mdp["states"] = states
    mdp["actions"] = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right
    mdp["goal_state"] = (3, 3, 0, 0)
    mdp["gamma"] = gamma
    mdp["theta"] = theta

    mdp["transitionProbabilities"] = {
        0: [0.8, 0.0, 0.1, 0.1],  # up:80%, left: 10%, right:10%
        1: [0.0, 0.8, 0.1, 0.1],  # down:80%, left: 10%, right:10%
        2: [0.1, 0.1, 0.8, 0.0],  # left:80%, up: 10%, down:10%
        3: [0.1, 0.1, 0.0, 0.8],  # right:80%, up: 10%, down:10%
    }

    return mdp

def get_rewards(state):
    x, y, water, fire = state
    if (x, y) == (3, 3):
        return 100  # goal state
    elif fire == 1:
        return -10  # fire penalty
    elif water == 1:
        return -5  # water penalty
    else:
        return -1  # movement cost

def get_next_states(state, action, mdp):
    next_states = {}
    x, y, water, fire = state
    for i in range(4):
        prob = mdp["transitionProbabilities"][action][i]
        if i == 0:  # up
            nx, ny = x, max(0, y - 1)
        elif i == 1:  # down
            nx, ny = x, min(3, y + 1)
        elif i == 2:  # left
            nx, ny = max(0, x - 1), y
        elif i == 3:  # right
            nx, ny = min(3, x + 1), y

        if (nx, ny) == (3, 3):
            newWater, newFire = 0, 0
        else:
            newWater = 1 if (nx, ny) in [(2, 1), (2, 2)] else 0
            newFire = 1 if (nx, ny) in [(0, 1), (0, 2)] else 0

        next_state = (nx, ny, newWater, newFire)

        if next_state in next_states:
            next_states[next_state] += prob
        else:
            next_states[next_state] = prob

    return next_states

# epsilon greedy and interact with the environment
def e_greedy(Q, state, epsilon, actions):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        else:
            q_values = Q[state]
            return max(q_values, key=q_values.get)
    
def interact(state, action, mdp):
        next_state_probs = get_next_states(state, action, mdp)
        next_states = list(next_state_probs.keys())
        probs = list(next_state_probs.values())
        next_states = random.choices(next_states, weights = probs, k = 1)[0]
        reward = get_rewards(next_states)
        return next_states, reward

# SARSA (TD)
def sarsa(mdp, alpha, gamma, epsilon, episodes = 100):
    # initialized Q
    Q = {}
    for s in mdp["states"]:
        Q[s] = {}
        for a in mdp["actions"]:
            Q[s][a] = 0

    returns_per_episode = []

    for ep in range(episodes):
        state = (0,0,0,0)
        action = e_greedy(Q, state, epsilon, mdp["actions"])
        total_reward = 0

        while state != mdp["goal_state"]:
            next_state, reward = interact(state, action, mdp)

            # epsilon greedy for next action
            next_action = e_greedy(Q, next_state, epsilon, mdp["actions"])
            
            # sarsa update
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            # update
            state = next_state
            action = next_action
            total_reward += reward

        returns_per_episode.append(total_reward)

    return Q, returns_per_episode

# Q learning, off policy, TD
def q_learning(mdp, alpha, gamma, epsilon, episodes = 100): 
    # initialize Q value
    Q = {}
    for s in mdp["states"]:
        Q[s] = {}
        for a in mdp["actions"]:
            Q[s][a] = 0
    
    returns_per_episode = []

    for i in range(episodes):
        state = (0, 0, 0, 0)
        total_reward = 0

        while state != mdp["goal_state"]:
            action = e_greedy(Q, state, epsilon, mdp["actions"])
            next_state, reward = interact(state, action, mdp)
            
            # Q learning update
            max_next_Q = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * max_next_Q - Q[state][action])

            # update 
            state = next_state
            total_reward += reward

        returns_per_episode.append(total_reward)

    return Q, returns_per_episode

# SARSA(lambda), backward eligibility traces
def sarsa_lambda(mdp, alpha, gamma, epsilon, _lambda, episodes = 100):
    # initialize Q value
    Q = {}
    for s in mdp["states"]:
        Q[s] = {}
        for a in mdp["actions"]:
            Q[s][a] = 0

    returns_per_episode = []

    for i in range(episodes):
        E = {}
        for s in mdp["states"]:
            E[s] = {}
            for a in mdp["actions"]:
                E[s][a] = 0

        state = (0, 0, 0, 0)
        action = e_greedy(Q, state, epsilon, mdp["actions"])
        total_reward = 0

        while state != mdp["goal_state"]:
            next_state, reward = interact(state, action, mdp)
            next_action = e_greedy(Q, next_state, epsilon, mdp["actions"])

            # TD Error
            delta = reward + gamma * Q[next_state][next_action] - Q[state][action]
            E[state][action] += 1 # accumulate trace

            for s in mdp["states"]:
                for a in mdp["actions"]:
                    Q[s][a] += alpha * delta * E[s][a]
                    E[s][a] *= gamma * _lambda # decay
            
            # update 
            state = next_state
            action = next_action
            total_reward += reward

        returns_per_episode.append(total_reward)

    return Q, returns_per_episode

def softmax(preference):
    max_pref = max(preference.values())
    
    exp_prefs = {}
    for a in preference:
        exp_prefs[a] = np.exp(preference[a] - max_pref)
    
    total = sum(exp_prefs.values())
    
    result = {}
    for a in preference:
        result[a] = exp_prefs[a] / total
    return result


def actor_critic_td_lambda(mdp, alpha_v = 0.1, alpha_pi = 0.01, gamma = 0.95, _lambda = 0.8, episodes = 100):
    # Initialize V 
    V = {}
    for s in mdp["states"]:
        V[s] = 0

    # Initialize theta
    theta = {}
    for s in mdp["states"]:
        theta[s] = {}
        for a in mdp["actions"]:
            theta[s][a] = 0

    returns_per_episode = []

    for i in range(episodes):
        state = (0, 0, 0, 0)
        total_reward = 0

        #initialized eligibility traces
        e_v = {}
        for s in mdp["states"]:
            e_v[s] = 0

        e_theta = {}
        for s in mdp["states"]:
            e_theta[s] = {}
            for a in mdp["actions"]:
                e_theta[s][a] = 0

        while state != mdp["goal_state"]:
            pi_s = softmax(theta[state])
            
            actions = list(pi_s.keys())
            probs = list(pi_s.values())
            # sample an action
            action = random.choices(actions, weights = probs, k = 1)[0]
            # interact
            next_state, reward = interact(state, action, mdp)
            # compute TD error for critic
            delta = reward + gamma * V[next_state] - V[state]

            # critic update
            e_v[state] = gamma * _lambda * e_v[state] + 1 # eligibility traces
            for s in e_v:
                if e_v[s] != 0:
                    V[s] += alpha_v * delta * e_v[s]
            
            # actor update
            policy_grad = {}
            for a in mdp["actions"]:
                if a == action:
                    policy_grad[a] = 1 - pi_s[a]
                else:
                    policy_grad[a] = -pi_s[a]
            
            for a in policy_grad:
                e_theta[state][a] = gamma * _lambda * e_theta[state][a] + policy_grad[a]

            # update theta
            for s in e_theta:
                for a in e_theta[s]:
                    if e_theta[s][a] != 0:
                        theta[s][a] += alpha_pi * delta * e_theta[s][a]
            # update state and reward
            state = next_state
            total_reward += reward
        
        returns_per_episode.append(total_reward)

    return V, theta, returns_per_episode

# plot single run of SARSA
def plot_sarsa_learning_curve(return_per_episode):

    plt.plot(return_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA Learning Curve')
    plt.show()

# plot single run of Q-learning
def plot_q_learning_curve(return_per_episode):

    plt.plot(return_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Learning Curve')
    plt.show()

# plot single run of SARSA(lambda)
def plot_sarsa_lambda_learning_curve(return_per_episode):

    plt.plot(return_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('SARSA(lambda) Learning Curve')
    plt.show()

def print_policy(Q, mdp, sample_size=10):

    print("\nSample of Learned Policy:")
    count = 0
    for state in mdp["states"]:
        if state == mdp["goal_state"]:
            continue
        best_action = max(Q[state], key=Q[state].get)
        print(f"State {state}: Best Action -> {best_action}")
        count += 1
        if count >= sample_size:
            break

def run_sarsa_once(mdp, alpha, gamma, epsilon, episodes = 100):
    Q, returns = sarsa(mdp, alpha, gamma, epsilon, episodes)
    return returns

def run_q_learning_once(mdp, alpha, gamma, epsilon, episodes = 100):
    Q, returns = q_learning(mdp, alpha, gamma, epsilon, episodes)
    return returns

def run_lambda_once(mdp, alpha, gamma, epsilon, _lambda, episodes = 100):
    Q, returns = sarsa_lambda(mdp, alpha, gamma, epsilon, _lambda, episodes)
    return returns

# SARSA mutiple run
def multiple_runs(mdp, alpha, gamma, epsilon, episodes = 100, trials = 100):
    all_returns = []
    for i in range(trials):
        returns = run_sarsa_once(mdp, alpha, gamma, epsilon, episodes)
        all_returns.append(returns)
        print("Trial", i+1, "/", trials, "completed")
    return np.array(all_returns)

def multiple_q_runs(mdp, alpha, gamma, epsilon, episodes = 100, trials = 100):
    all_returns = []
    for i in range(trials):
        returns = run_q_learning_once(mdp, alpha, gamma, epsilon, episodes)
        all_returns.append(returns)
        print("Trial", i+1, "/", trials, "completed")
    return np.array(all_returns)

def multiple_lambda_runs(mdp, alpha, gamma, epsilon, _lambda, episodes = 100, trials = 100):
    all_returns = []
    for i in range(trials):
        returns = run_lambda_once(mdp, alpha, gamma, epsilon, _lambda, episodes)
        all_returns.append(returns)
        print("Trial", i+1, "/", trials, "completed")
    return np.array(all_returns)

# plot shade, min-max range 
def plot_sarsa_with_min_max(all_returns):
    mean_returns = np.mean(all_returns, axis = 0)
    max_returns = np.max(all_returns, axis = 0)
    min_returns = np.min(all_returns, axis = 0)
    
    plt.figure(figsize = (10, 6))
    plt.plot(mean_returns, label = 'average total reward')
    plt.fill_between(range(len(mean_returns)), min_returns, max_returns, color = 'red', alpha = 0.2, label = 'Min-Max range')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('SARSA Learning Curve')
    plt.legend()
    plt.show()

def plot_q_with_min_max(all_returns):
    mean_returns = np.mean(all_returns, axis = 0)
    max_returns = np.max(all_returns, axis = 0)
    min_returns = np.min(all_returns, axis = 0)
    
    plt.figure(figsize = (10, 6))
    plt.plot(mean_returns, label = 'average total reward')
    plt.fill_between(range(len(mean_returns)), min_returns, max_returns, color = 'blue', alpha = 0.2, label = 'Min-Max range')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Q-learning Learning Curve')
    plt.legend()
    plt.show()

def plot_sarsa_lambda_with_min_max(all_returns):
    mean_returns = np.mean(all_returns, axis = 0)
    max_returns = np.max(all_returns, axis = 0)
    min_returns = np.min(all_returns, axis = 0)
    
    plt.figure(figsize = (10, 6))
    plt.plot(mean_returns, label = 'average total reward')
    plt.fill_between(range(len(mean_returns)), min_returns, max_returns, color = 'orange', alpha = 0.2, label = 'Min-Max range')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('SARSA(lambda) Learning Curve')
    plt.legend()
    plt.show()

def plot_actor_critic_with_min_max(all_returns):
    mean_returns = np.mean(all_returns, axis = 0)
    max_returns = np.max(all_returns, axis = 0)
    min_returns = np.min(all_returns, axis = 0)
    
    plt.figure(figsize = (10, 6))
    plt.plot(mean_returns, label = 'average total reward', linewidth = 2)
    plt.fill_between(range(len(mean_returns)), min_returns, max_returns, color = 'green', alpha = 0.2, label = 'Min-Max range')
    plt.xlabel('Episode')
    plt.ylabel('Total Rewards')
    plt.title('Actor-Critic with TD(lambda) Learning Curve')
    plt.legend()
    plt.show()

# main function 
def main_sarsa():
    mdp = build_mdp(gamma=0.95)
    # hyperparameter
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    episodes = 100
    trials = 100

    # print hyperparameter and initial policy
    print("Hyperparameters used: alpha = ", alpha, "epsilon =", epsilon, "gamma =", gamma)
    print("Initial policy: Random (epsilon-greedy) with Q-values initialized to 0")
    # run multiple trials and collect returns of each episode
    all_returns = multiple_runs(mdp, alpha, gamma, epsilon, episodes, trials)
    plot_sarsa_with_min_max(all_returns)
    
    Q, return_per_episode = sarsa(mdp, alpha, gamma, epsilon, episodes)

    print_policy(Q, mdp)
    # plot final Q learning curve
    plot_sarsa_learning_curve(return_per_episode)
    
    last_episode_rewards = all_returns[:, -1]
    average_last_reward = np.mean(last_episode_rewards)
    std_last_reward = np.std(last_episode_rewards)

    print("SARSA - Average total reward is", average_last_reward)
    print("SARSA - Standard deviation is", std_last_reward)
    return

def main_q_learning():
    mdp = build_mdp(gamma=0.95)
    # hyperparameter
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    episodes = 100
    trials = 100
    
    # print hyperparameter and initial policy
    print("Q-learning")
    print("Hyperparameters used: alpha =", alpha, "epsilon =", epsilon, "gamma =", gamma)
    print("Initial policy: Random (epsilon-greedy) with Q-values initialized to 0")
    
    # run multiple trials and collect returns of each episode
    all_returns = multiple_q_runs(mdp, alpha, gamma, epsilon, episodes, trials)
    plot_q_with_min_max(all_returns)

    # run single Q learning
    Q, return_per_episode = q_learning(mdp, alpha, gamma, epsilon, episodes)

    print_policy(Q, mdp)

    plot_q_learning_curve(return_per_episode)
    last_episode_rewards = all_returns[:, -1]
    average_last_reward = np.mean(last_episode_rewards)
    std_last_reward = np.std(last_episode_rewards)

    # print final statistic outcome
    print("Q-learning - Average total reward at episode 100:", average_last_reward)
    print("Q-learning - Standard deviation at episode 100:", std_last_reward)
    return

def main_sarsa_lambda():
    mdp = build_mdp(gamma=0.95)
    # hyperparameter
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.95
    _lambda = 0.8
    episodes = 100
    trials = 100

    # print hyperparameter and initial policy
    print("Hyperparameters used: alpha =", alpha, "epsilon =", epsilon, "gamma =", gamma, "lambda =", _lambda)
    print("Initial policy: Random (epsilon-greedy) with Q-values initialized to 0")

    all_returns = multiple_lambda_runs(mdp, alpha, gamma, epsilon, _lambda, episodes, trials)
    plot_sarsa_lambda_with_min_max(all_returns)

    Q, return_per_episode = sarsa_lambda(mdp, alpha, gamma, epsilon, _lambda, episodes)

    print_policy(Q, mdp)

    plot_sarsa_lambda_learning_curve(return_per_episode)
    last_episode_rewards = all_returns[:, -1]
    average_last_reward = np.mean(last_episode_rewards)
    std_last_reward = np.std(last_episode_rewards)

    # print final statistic outcome
    print("SARSA(λ) - Average total reward at episode 100:", average_last_reward)
    print("SARSA(λ) - Standard deviation at episode 100", std_last_reward)
    return

def main_actor_critic():
    mdp = build_mdp(gamma=0.95)
    # Hyperparameter
    alpha_v = 0.1
    alpha_pi = 0.01
    epsilon = 0.1
    gamma = 0.95
    _lambda = 0.8
    episodes = 100
    trials = 100

    # print hyperparameter and initial policy
    print("Hyperparameters used: alpha_v =", alpha_v, "alpha_pi =", alpha_pi, "gamma =", gamma, "lambda =", _lambda)
    print("Initial policy: Uniform softmax with theta initialized to 0")

    # run multiple and collect returns from each trial
    all_returns = []
    for i in range(trials):
        V, theta, returns = actor_critic_td_lambda(mdp, alpha_v, alpha_pi, gamma, _lambda, episodes)
        all_returns.append(returns)
        print("Trial", str(i + 1) + "/" + str(trials), "completed")

    all_returns = np.array(all_returns)

    plot_actor_critic_with_min_max(all_returns)
    last_episode_rewards = all_returns[:, -1]
    average_last_reward = np.mean(last_episode_rewards)
    std_last_reward = np.std(last_episode_rewards)

    # print final statistic outcome
    print("Actor-Critic - Average total reward at episode 100:", average_last_reward)
    print("Actor-Critic - Standard deviation at episode 100:", std_last_reward)
    return

if __name__ == "__main__":
    main_sarsa()
    main_q_learning()
    main_sarsa_lambda()
    main_actor_critic()