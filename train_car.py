import enviroment as env
from xdsDQN import DeepQNetwork

env.main()
RL = DeepQNetwork(n_actions=8,n_features=env.re_image(self=1).shape[0],learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, )

total_steps = 0
for i_episode in range(10000):

    # 获取回合 i_episode 第一个 observation
    observation = env.re_image()
    ep_r = 0

    while True:
        # env.render()    # 刷新环境
        print(1)
        action = RL.choose_action(observation)  # 选行为
        env.send_control(action)

        # observation_, reward, done, info = env.step(action) # 获取下一个 state
        observation_=env.re_image()


        reward,done =env.re_rdanddone()
        #保存这一组记忆
        RL.store_transition(observation, action, reward, observation_)
        #
        if total_steps > 1000:
            RL.learn(do_train=1)  # 学习

        ep_r += reward
        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

# 最后输出 cost 曲线
RL.plot_cost()