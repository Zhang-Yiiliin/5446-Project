import argparse
import torch
import numpy as np
import gym
import pygame as pg
import os, sys

from hyperparameters import get_hyperparameters
from wrappers import wrap_deepmind, wrap_pytorch, make_atari

def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('dqn_type', type=str, help='[dqn, double, dueling, prioritize, noisy]')
    parser.add_argument('game', type=str, help='[cartpole, breakout, pong]')
    args = parser.parse_args()
    return args.dqn_type, args.game

def run(env, model):
    state, _ = env.reset()
    while True:
        env.render()
        action = model.act(state, 0)
        state, _, terminated, _, _ = env.step(action)
        if terminated or pg.key.get_pressed()[pg.K_ESCAPE]:
            break

def main():
    dqn_type, game = parse_input()
    Model, _, _, _, _, _, _, _, _, _ = get_hyperparameters(dqn_type, game)
    if game == "cartpole":
        env = gym.make("CartPole-v1", render_mode='human')
        model = Model(env.observation_space.shape[0], env.action_space.n)
        if os.path.exists(f"results/{dqn_type}_{game}/model.pt"):
            model.load_state_dict(torch.load(f"results/{dqn_type}_{game}/model.pt"))
            run(env, model)
        else:
            print("model not trained", file=sys.stderr)
    elif game == "breakout":
        env = wrap_pytorch(wrap_deepmind(make_atari("BreakoutNoFrameskip-v4", render_mode='human')))
        model = Model(env.observation_space.shape, env.action_space.n)
        if os.path.exists(f"results/{dqn_type}_{game}/model.pt"):
            model.load_state_dict(torch.load(f"results/{dqn_type}_{game}/model.pt"))
            run(env, model)
        else:
            print("model not trained", file=sys.stderr)
    elif game == "pong":
        env = wrap_pytorch(wrap_deepmind(make_atari("PongNoFrameskip-v4", render_mode='human')))
        model = Model(env.observation_space.shape, env.action_space.n)
        if os.path.exists(f"results/{dqn_type}_{game}/model.pt"):
            model.load_state_dict(torch.load(f"results/{dqn_type}_{game}/model.pt"))
            run(env, model)
        else:
            print("model not trained", file=sys.stderr)

if __name__ == "__main__":
    main()