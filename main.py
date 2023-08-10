import os
import joblib
import numpy as np
import gym

from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env_name = 'LunarLanderContinuous-v2'
    env = gym.make(env_name)

    agent = Agent(alpha=0.0003, beta=0.0003, 
                  state_dims=env.observation_space.shape[0], tau=0.005,
                  env=env, batch_size=256, layer1_size=256, layer2_size=256,
                  action_dims=env.action_space.shape[0], entropy_coef=0.2)
    
    n_games = 2000
    filename = env_name + '_'+ str(n_games) + 'entropy_coef' + \
        str(agent.entropy_coeficient) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, _, _ = env.step(action)
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            score += reward
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        print('episode ', i, 'score %.2f' % score,
                'trailing 100 games avg %.3f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    joblib.dump(score_history, os.path.join('checkpoint', 'score_history'))
