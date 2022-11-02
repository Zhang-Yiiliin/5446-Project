import argparse
import gym
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from hyperparameters import get_hyperparameters
from loss import td_loss


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('dqn_type', type=str, help='[dqn, double, dueling, prioritize, noisy]')
    parser.add_argument('game', type=str, help='[cartpole, breakout, pong]')
    args = parser.parse_args()
    return args.dqn_type, args.game


def train(env, num_frames, batch_size, train_initial, gamma, epsilon_func, model, buffer, optimizer, loss_func):
    losses = []
    all_rewards = []

    episode_reward = 0
    state, _ = env.reset()
    for frame_idx in tqdm(range(1, num_frames + 1)):
        # select an action
        epsilon = epsilon_func(frame_idx)
        action = model.act(state, epsilon)
    
        # save transition into buffer
        next_state, reward, done, _, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            # reset the environment, record total reward
            state, _ = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
  
        if len(buffer) >= train_initial:
            # start training after collected enough samples
            loss = loss_func(batch_size, gamma, model, buffer, optimizer)
            losses.append(loss.detach().item())

    return losses, all_rewards


def plot(losses, rewards, path):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('reward')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(path)


def main():
    dqn_type, game = parse_input()
    Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial = get_hyperparameters(dqn_type, game)
    if game == "cartpole":
        env = gym.make("CartPole-v1")
        model = Model(env.observation_space.shape[0], env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        replay_buffer = Buffer(buffer_size)

    elif game == "pong" and dqn_type == "dqn":
        env = wrap_pytorch(wrap_deepmind(make_atari("PongNoFrameskip-v4")))
        model = Model(env.observation_space.shape, env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        replay_buffer = Buffer(buffer_size)

    elif game == "breakout" and dqn_type == "dqn":
        env = wrap_pytorch(wrap_deepmind(make_atari("BreakoutNoFrameskip-v4")))
        model = Model(env.observation_space.shape, env.action_space.n)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        replay_buffer = Buffer(buffer_size)

    losses, rewards = train(env, num_frames, batch_size, train_initial, gamma, epsilon_func, model, replay_buffer, optimizer, loss_func)
    losses, rewards = np.array(losses), np.array(rewards)

    # save result
    if not os.path.exists(f"results/{dqn_type}_{game}"):
        os.mkdir(f"results/{dqn_type}_{game}")
    plot(losses, rewards, f"results/{dqn_type}_{game}/reward_and_loss.png")
    torch.save(model.state_dict, f'results/{dqn_type}_{game}/model.pt')
    np.savez(f"results/{dqn_type}_{game}/loss_and_reward.npz", loss=losses, reward=rewards)

if __name__ == "__main__":
    main()
