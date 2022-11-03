# 5446-Project
CS5446 AI Planning and Decision Making

1. Vanilla DQN

2. Double DQN

3. Dueling DQN

4. Prioritized DQN

5. Noisy DQN

**install requirements**
```
python -m pip install -r requirements.txt
```

**example usage:**

train agent play cartpole game with vanilla dqn
```
python train.py dqn cartpole
```
evaluate the result
```
python evaluate.py dqn cartpole
```

**other choices**

dqn_type = [dqn, double, dueling, prioritized, noisy]

game = [cartpole, pong, breakout]