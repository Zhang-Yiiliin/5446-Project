import math

from models.mlp import MLP_DQN
from models.cnn import CNN_DQN
from models.dueling import Dueling_DQN
from models.noisy import Noisy_CNN_DQN
from replaybuffer.randombuffer import RandomBuffer
from replaybuffer.prioritizedbuffer import PrioritizedBuffer
from loss import td_loss, double_td_loss, prioritized_td_loss

def get_hyperparameters(dqn_type, game):
    """
    return: Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial, 
    """
    hyperparameters = dict()
    if dqn_type == "dqn" and game == "cartpole":
        hyperparameters["Model"] = MLP_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: 0.01 + (1.0 - 0.01) * math.exp(-1. * x / 500)
        hyperparameters["loss_func"] = td_loss
        hyperparameters["num_frames"] = 10000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.001
        hyperparameters["buffer_size"] = 1000
        hyperparameters["train_initial"] = 1000
    elif game == "cartpole":
        raise NotImplementedError(f"{dqn_type} dqn for cartpole is not supported")
    elif dqn_type == "dqn" and game == "pong":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = td_loss
        hyperparameters["num_frames"] = 1000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.00001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "dqn" and game == "breakout":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = td_loss
        hyperparameters["num_frames"] = 3000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.00001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "double" and game == "pong":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 1000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.00001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "double" and game == "breakout":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 3000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.00001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "dueling" and game == "pong":
        hyperparameters["Model"] = Dueling_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 1000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "dueling" and game == "breakout":
        hyperparameters["Model"] = Dueling_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 3000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "prioritized" and game == "pong":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = PrioritizedBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["beta_func"] = lambda frame_idx: min(1.0, 0.4 + frame_idx * (1.0 - 0.4) / 100000)
        hyperparameters["loss_func"] = prioritized_td_loss
        hyperparameters["num_frames"] = 1000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "prioritized" and game == "breakout":
        hyperparameters["Model"] = CNN_DQN
        hyperparameters["Buffer"] = PrioritizedBuffer
        hyperparameters["epsilon_func"] = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        hyperparameters["beta_func"] = lambda frame_idx: min(1.0, 0.4 + frame_idx * (1.0 - 0.4) / 100000)
        hyperparameters["loss_func"] = prioritized_td_loss
        hyperparameters["num_frames"] = 3000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "noisy" and game == "pong":
        hyperparameters["Model"] = Noisy_CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 1000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    elif dqn_type == "noisy" and game == "breakout":
        hyperparameters["Model"] = Noisy_CNN_DQN
        hyperparameters["Buffer"] = RandomBuffer
        hyperparameters["loss_func"] = double_td_loss
        hyperparameters["num_frames"] = 3000000
        hyperparameters["gamma"] = 0.99
        hyperparameters["batch_size"] = 32
        hyperparameters["lr"] = 0.0001
        hyperparameters["buffer_size"] = 100000
        hyperparameters["train_initial"] = 10000
    hyperparameters["dqn_type"] = dqn_type
    hyperparameters["game"] = game
    return hyperparameters
