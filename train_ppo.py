# import vessl
from cfg import *
from agent.ppo import *  # 학습 알고리즘
from environment.env import WeldingLine  # 학습 환경

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



if __name__ == "__main__":
    cfg = get_cfg()
    # vessl.init(organization="snu-eng-dgx", project="SHI-FINAL", hp=cfg)

    load_model = False
    num_block = 80
    weight_list = [1.0, 0.0]

    rule_weight = {
        60: {"ATCS": [5.535, 0.943], "COVERT": 0.4},
        80: {"ATCS": [5.953, 1.000], "COVERT": 6.605},
        100: {"ATCS": [6.122, 0.953], "COVERT": 9.0},
        160: {"ATCS": [6.852, 1.173], "COVERT": 1.4},
        240: {"ATCS": [7.482, 1.482], "COVERT": 0.9}}

    for weight in weight_list:
        weight_tard = weight
        weight_setup = 1 - weight
        learning_rate = 0.005 if not cfg.use_vessl else cfg.lr
        K_epoch = 1 if not cfg.use_vessl else cfg.K_epoch
        T_horizon = "entire" if not cfg.use_vessl else cfg.T_horizon

        num_episode = 10000

        with open('SHI_sample.json', 'r') as f:
            block_sample = json.load(f)

        model_dir = './output/lr_{0}_K_{1}_T_{2}/{3}_{4}/model/'.format("5e-3", K_epoch, T_horizon, round(10 * weight), 10 - round(10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/model/".format(round(10 * weight), 10 - round(10 * weight))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        simulation_dir = './output/lr_{0}_K_{1}_T_{2}/{3}_{4}/simulation/'.format("5e-3", K_epoch, T_horizon, round(10 * weight), 10 - round(10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/simulation/".format(round(10 * weight), 10 - round(10 * weight))
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        log_dir = './output/lr_{0}_K_{1}_T_{2}/{3}_{4}/log/'.format("5e-3", K_epoch, T_horizon, round(10 * weight), 10 - round(10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/log/".format(round(10 * weight), 10 - round(10 * weight))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = WeldingLine(num_block=num_block, reward_weight=[weight_tard, weight_setup], block_sample=block_sample,
                          rule_weight=rule_weight[num_block], is_train=True)

        agent = PPO(env.state_size, env.action_size, learning_rate, cfg.gamma, cfg.lmbda, cfg.eps_clip, K_epoch, T_horizon).to(device)

        # if cfg.load_model:
        # checkpoint = torch.load('./output/train_ddt/model/episode-10001.pt')
        # start_episode = checkpoint['episode'] + 1
        # agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # # else:
        start_episode = 1

        with open(log_dir + "train_log.csv", 'w') as f:
            f.write('episode, reward, reward_tard, reward_setup\n')
        # with open(log_dir + "validation_log.csv", 'w') as f:
        #     f.write('episode, tardiness, setup_ratio, makespan\n')

        for episode in range(start_episode, start_episode + num_episode + 1):
            state = env.reset()
            r_epi = 0.0
            done = False

            while not done:
                # for t in range(T_horizon):
                logit = agent.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done = env.step(action)

                agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    # print("episode: %d | reward: %.4f" % (e, r_epi))
                    # print("episode: %d | total_rewards: %.2f" % (episode, r_epi))
                    tardiness = (np.sum(env.monitor.tardiness) / env.num_block) * 24
                    setup = env.monitor.setup / env.model["Sink"].total_finish
                    # makespan = max(test_env.monitor.throughput.keys())
                    makespan = env.model["Sink"].makespan
                    print("episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f" % (
                        episode, r_epi, setup, tardiness, makespan))
                    with open(log_dir + "train_log.csv", 'a') as f:
                        f.write('%d,%.2f,%.2f,%.2f\n' % (episode, r_epi, env.tard_reward, env.setup_reward))
                    break
            agent.train_net()

            if episode % 1000 == 0 or episode == 1:
                # tardiness = (np.sum(env.monitor.tardiness) / env.num_block) * 24
                # setup = np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish
                # # makespan = max(test_env.monitor.throughput.keys())
                # makespan = env.model["Sink"].makespan

                # vessl.log(step=episode, payload={'reward': r_epi})
                # vessl.log(step=episode, payload={'reward_setup': env.setup_reward})
                # vessl.log(step=episode, payload={'reward_tard': env.tard_reward})
                # vessl.log(step=episode, payload={'Train_Tardiness': tardiness})
                # vessl.log(step=episode, payload={'Train_Setup': setup})
                # vessl.log(step=episode, payload={'Train_Makespan': makespan})

                # _ = env.get_logs(simulation_dir + "/log_{0}.csv".format(episode))
                agent.save(episode, model_dir)

            # if episode % 100 == 1:
            #     test_tardiness, test_setup_ratio, test_makespan = evaluate(episode, agent, simulation_dir, test_sample_data)
            #     with open(log_dir + "validation_log.csv", 'a') as f:
            #         f.write('%d,%.4f,%.4f,%.4f \n' % (episode, test_tardiness, test_setup_ratio, test_makespan))
                #
                # vessl.log(step=episode, payload={'Test_Tardiness': test_tardiness})
                # vessl.log(step=episode, payload={'Test_Setup': test_setup_ratio})
                # vessl.log(step=episode, payload={'Test_Makespan': test_makespan})

