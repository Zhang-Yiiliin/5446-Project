import torch
import numpy as np

def td_loss(num_frames, batch_size, gamma, model, buffer, optimizer):
    state, action, reward, next_state, done = buffer.sample(batch_size)

    state      = torch.FloatTensor(np.float32(state))
    next_state = torch.FloatTensor(np.float32(next_state))
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)

    if torch.cuda.is_available():
        state, next_state, action, reward, done = state.cuda(), next_state.cuda(), action.cuda(), reward.cuda(), done.cuda()

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) # get q_value indexed by action
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done) # if done = 1, no next state
    
    loss = (q_value - expected_q_value.data).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


def double_td_loss(num_frames, batch_size, gamma, model, tmodel, buffer, optimizer):
    state, action, reward, next_state, done = buffer.sample(batch_size)

    state      = torch.FloatTensor(np.float32(state))
    next_state = torch.FloatTensor(np.float32(next_state))
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)

    if torch.cuda.is_available():
        state, next_state, action, reward, done = state.cuda(), next_state.cuda(), action.cuda(), reward.cuda(), done.cuda()

    q_values      = model(state)
    next_q_values = model(next_state)
    next_q_state_values = tmodel(next_state) 

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value     = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.data).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()


def prioritized_td_loss(num_frames, batch_size, gamma, model, tmodel, buffer, optimizer):
    beta = buffer.beta_func(num_frames)
    state, action, reward, next_state, done, indices, weights = buffer.sample(batch_size, beta)

    state      = torch.FloatTensor(np.float32(state))
    next_state = torch.FloatTensor(np.float32(next_state))
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)
    weights    = torch.FloatTensor(weights)

    if torch.cuda.is_available():
        state, next_state, action, reward, done, weights = state.cuda(), next_state.cuda(), action.cuda(), reward.cuda(), done.cuda(), weights.cuda()

    q_values      = model(state)
    next_q_values = model(next_state)
    next_q_state_values = tmodel(next_state) 

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1) 
    next_q_value     = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss  = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss  = loss.mean()
    buffer.update_priorities(indices, prios.data.cpu().numpy())
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach()
