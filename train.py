import argparse
import gym
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from hyperparameters import get_hyperparameters

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('dqn_type', type=str, help='[dqn, double, dueling, prioritize, noisy]')
    parser.add_argument('game', type=str, help='[cartpole, breakout, pong]')
    args = parser.parse_args()
    return args.dqn_type, args.game


def train(env, num_frames, batch_size, train_initial, gamma, epsilon_func, model, tmodel, buffer, optimizer, loss_func):
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
            if tmodel:
                loss = loss_func(num_frames, batch_size, gamma, model, tmodel, buffer, optimizer)
            else:
                loss = loss_func(num_frames, batch_size, gamma, model, buffer, optimizer)
            losses.append(loss.detach().item())

        if num_frames % 1000 == 0 and tmodel:
            tmodel.load_state_dict(model.state_dict())

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
        # demo
        env = gym.make("CartPole-v1")
        model = Model(env.observation_space.shape[0], env.action_space.n)
        tmodel = None
    else:
        # select game
        if game == "pong":
            env = wrap_pytorch(wrap_deepmind(make_atari("PongNoFrameskip-v4")))
        elif game == "breakout":
            env = wrap_pytorch(wrap_deepmind(make_atari("BreakoutNoFrameskip-v4")))
        else:
            print(f"{game} not supported", file=sys.stderr)
        # select model
        if dqn_type == "dqn":
            model = Model(env.observation_space.shape, env.action_space.n)
            tmodel = None
            if torch.cuda.is_available():
                model = model.cuda()
            replay_buffer = Buffer(buffer_size)
        elif dqn_type == "double":
            model, tmodel = Model(env.observation_space.shape, env.action_space.n), Model(env.observation_space.shape, env.action_space.n)
            tmodel.load_state_dict(model.state_dict())
            if torch.cuda.is_available():
                model, tmodel = model.cuda(), tmodel.cuda()
            replay_buffer = Buffer(buffer_size)
        elif dqn_type == "dueling":
            # use double td_loss train dueling dqn
            model, tmodel = Model(env.observation_space.shape, env.action_space.n), Model(env.observation_space.shape, env.action_space.n)
            tmodel.load_state_dict(model.state_dict())
            if torch.cuda.is_available():
                model, tmodel = model.cuda(), tmodel.cuda()
            replay_buffer = Buffer(buffer_size)
        elif dqn_type == "prioritized":
            model, tmodel = Model(env.observation_space.shape, env.action_space.n), Model(env.observation_space.shape, env.action_space.n)
            tmodel.load_state_dict(model.state_dict())
            if torch.cuda.is_available():
                model, tmodel = model.cuda(), tmodel.cuda()
            beta_func = lambda frame_idx: min(1.0, 0.4 + frame_idx * (1.0 - 0.4) / 100000)
            replay_buffer = Buffer(buffer_size, beta_func)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses, rewards = train(env, num_frames, batch_size, train_initial, gamma, epsilon_func, model, tmodel, replay_buffer, optimizer, loss_func)
    losses, rewards = np.array(losses), np.array(rewards)

    # save result
    if not os.path.exists(f"results/{dqn_type}_{game}"):
        os.mkdir(f"results/{dqn_type}_{game}")
    plot(losses, rewards, f"results/{dqn_type}_{game}/reward_and_loss.png")
    torch.save(model.state_dict(), f'results/{dqn_type}_{game}/model.pt')
    np.savez(f"results/{dqn_type}_{game}/loss_and_reward.npz", loss=losses, reward=rewards)

if __name__ == "__main__":
    main()
