from xdsnewDQN import DeepQNetwork


def rl_brain():
    RL = DeepQNetwork(n_actions=7,
                      n_features=100800,
                      learning_rate=0.0001, e_greedy=0.9,
                      replace_target_iter=1000, memory_size=2000,
                      e_greedy_increment=0.000001, )

    total_steps = 0
    ep_r = 0
    return RL,total_steps,ep_r