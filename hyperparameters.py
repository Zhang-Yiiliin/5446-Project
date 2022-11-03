import math

from models.mlp import MLP_DQN
from models.cnn import CNN_DQN
from models.dueling import Dueling_DQN
from replaybuffer.randombuffer import RandomBuffer
from loss import td_loss, double_td_loss

def get_hyperparameters(dqn_type, game):
    """
    return: Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial, 
    """
    if dqn_type == "dqn" and game == "cartpole":
        Model = MLP_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: 0.01 + (1.0 - 0.01) * math.exp(-1. * x / 500)
        loss_func = td_loss
        num_frames = 10000
        gamma = 0.99
        batch_size = 32
        lr = 0.001
        buffer_size = 1000
        train_initial = 1000
        return Model, Buffer, epsilon_func, td_loss, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif game == "cartpole":
        raise NotImplementedError(f"{dqn_type} dqn for cartpole is not supported")
    elif dqn_type == "dqn" and game == "pong":
        Model = CNN_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = td_loss
        num_frames = 1000000
        gamma = 0.99
        batch_size = 32
        lr = 0.00001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif dqn_type == "dqn" and game == "breakout":
        Model = CNN_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = td_loss
        num_frames = 3000000
        gamma = 0.99
        batch_size = 32
        lr = 0.00001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif dqn_type == "double" and game == "pong":
        Model = CNN_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = double_td_loss
        num_frames = 1000000
        gamma = 0.99
        batch_size = 32
        lr = 0.00001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif dqn_type == "double" and game == "breakout":
        Model = CNN_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = double_td_loss
        num_frames = 3000000
        gamma = 0.99
        batch_size = 32
        lr = 0.00001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif dqn_type == "dueling" and game == "pong":
        Model = Dueling_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = double_td_loss
        num_frames = 1000000
        gamma = 0.99
        batch_size = 32
        lr = 0.0001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
    elif dqn_type == "dueling" and game == "breakout":
        Model = Dueling_DQN
        Buffer = RandomBuffer
        epsilon_func = lambda x: max(1. * (0.999 ** (x // 500)), 0.01)
        loss_func = double_td_loss
        num_frames = 3000000
        gamma = 0.99
        batch_size = 32
        lr = 0.0001
        buffer_size = 100000
        train_initial = 10000
        return Model, Buffer, epsilon_func, loss_func, num_frames, gamma, batch_size, lr, buffer_size, train_initial
