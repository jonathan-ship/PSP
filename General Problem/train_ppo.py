import os
import vessl
import torch
import json
import math

from cfg import get_cfg
from agent.ppo import *
from environment.env import PMSP

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.use_vessl:
        vessl.init(organization="sun-eng-dgx", project="Final-General-PMSP", hp=cfg)

    rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
                   200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
                   400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    load_model = False
    weight_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]
    for weight in weight_list:
        weight_tard = weight
        weight_setup = 1 - weight
        optim = "Adam"
        learning_rate = 0.005 if not cfg.use_vessl else cfg.lr
        K_epoch = 1 if not cfg.use_vessl else cfg.K_epoch
        T_horizon = "entire" if not cfg.use_vessl else cfg.T_horizon
        num_episode = 10000
        num_job = cfg.num_job
        num_m = cfg.num_machine

        model_dir = './output/lr_{0}_K_{1}_T_{2}_2/{3}_{4}/model/'.format("5e-3", K_epoch, T_horizon, round(10 * weight),
                                                                        10 - round(
                                                                            10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/model/".format(
            round(10 * weight), 10 - round(10 * weight))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        simulation_dir = './output/lr_{0}_K_{1}_T_{2}_2/{3}_{4}/simulation/'.format("5e-3", K_epoch, T_horizon,
                                                                                  round(10 * weight), 10 - round(
                10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/simulation/".format(round(10 * weight),
                                                                                             10 - round(10 * weight))
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        log_dir = './output/lr_{0}_K_{1}_T_{2}_2/{3}_{4}/log/'.format("5e-3", K_epoch, T_horizon, round(10 * weight),
                                                                    10 - round(
                                                                        10 * weight)) if not cfg.use_vessl else "/output/{0}_{1}/log/".format(
            round(10 * weight), 10 - round(10 * weight))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        env = PMSP(num_job=num_job, num_m=num_m, reward_weight=[weight_tard, weight_setup], rule_weight=rule_weight[num_job])
        agent = PPO(cfg, env.state_dim, env.action_dim, optimizer_name=optim, K_epoch=K_epoch).to(device)

        # if cfg.load_model:
        #     checkpoint = torch.load('./trained_model/episode-30000.pt')
        #     start_episode = checkpoint['episode'] + 1
        #     agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # else:
        start_episode = 1

        with open(log_dir + "train_log.csv", 'w') as f:
            f.write('episode, reward, reward_tard, reward_setup \n')

        for episode in range(start_episode, start_episode + num_episode):
            state = env.reset()
            r_epi = 0.0
            done = False

            while not done:
                # for t in range(cfg.T_horizon):
                logit = agent.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done = env.step(action)

                agent.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    tardiness = env.monitor.tardiness / env.num_job
                    setup = env.monitor.setup / env.num_job
                    makespan = env.sink.makespan
                    print("{0}_{1} |".format(int(10 * weight), int(10 * (1 - weight))),
                          "episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | makespan %.4f" % (
                          episode, r_epi, setup, tardiness, makespan))
                    with open(log_dir + "train_log.csv", 'a') as f:
                        f.write('%d,%.2f,%.2f,%.2f\n' % (
                        episode, r_epi, env.reward_tard, env.reward_setup))
                    break
            agent.train_net()

            if episode % 100 == 0 or episode == 1:
                # _ = env.get_logs(simulation_dir + "log_{0}.csv".format(episode))
                agent.save(episode, model_dir)
                if cfg.use_vessl:
                    tardiness = env.monitor.tardiness / env.num_job
                    setup = env.monitor.setup / env.num_job
                    vessl.log(step=episode, payload={'reward{0}{1}'.format(round(10 * weight), 10-round(10*weight)): r_epi})
                    vessl.log(step=episode, payload={'reward_setup{0}{1}'.format(round(10 * weight), 10-round(10*weight)): env.reward_setup})
                    vessl.log(step=episode, payload={'reward_tard{0}{1}'.format(round(10 * weight), 10-round(10*weight)): env.reward_tard})
                    vessl.log(step=episode, payload={'Train_Tardiness{0}{1}'.format(round(10 * weight), 10-round(10*weight)): tardiness})
                    vessl.log(step=episode, payload={'Train_Setup{0}{1}'.format(round(10 * weight), 10-round(10*weight)): setup})