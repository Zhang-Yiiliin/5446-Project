import argparse
import imageio
import numpy as np
import os
from PIL import Image
import PIL.ImageDraw as ImageDraw
import torch
import time

from wrappers import make_atari, wrap_deepmind, wrap_pytorch
from hyperparameters import get_hyperparameters


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('dqn_type', type=str, help='[dqn, double, dueling, prioritize, noisy]')
    parser.add_argument('game', type=str, help='[breakout, pong]')
    # parser.add_argument('num_frame', type=str, help='n % 50000 == 0')
    args = parser.parse_args()
    return args.dqn_type, args.game # args.num_frame


def _label_with_num_frame(frame, num_frame):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)

    drawer.text((im.size[0]/20,im.size[1]/18), f'trained frames: {num_frame}', fill=text_color)
    return im


def run(dqn_type, game, env, model, num_frames):
    state, _ = env.reset()
    frames = []
    for num_frame in num_frames:
        model.load_state_dict(torch.load(f"modelstats/{dqn_type}_{game}/{game}_{num_frame}_frame_{dqn_type}.pt", map_location=torch.device('cpu')))
        
        while True:
            frame = env.render()
            frames.append(_label_with_num_frame(frame, num_frame))

            action = model.act(state, 0)
            state, _, terminated, _, _ = env.step(action)
            if terminated:
                state, _ = env.reset()
            if len(frames) % 480 == 0:
                break

    imageio.mimwrite(f'gifs/{dqn_type}_{game}.gif', frames, fps=60)
    env.close()


def main():
    dqn_type, game = parse_input()
    Model, _, _, _, _, _, _, _, _, _ = get_hyperparameters(dqn_type, game)
    if game == "breakout":
        env = wrap_pytorch(wrap_deepmind(make_atari("BreakoutNoFrameskip-v4")))
        model = Model(env.observation_space.shape, env.action_space.n)
        num_frames = [50000, 250000, 500000, 1500000, 3000000]
    elif game == "pong":
        env = wrap_pytorch(wrap_deepmind(make_atari("PongNoFrameskip-v4")))
        model = Model(env.observation_space.shape, env.action_space.n)
        num_frames = [50000, 100000, 200000, 500000, 1000000]
    else:
        raise NotImplementedError()

    run(dqn_type, game, env, model, num_frames)


if __name__ == "__main__":
    main()