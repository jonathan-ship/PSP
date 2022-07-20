# import vessl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical

# from environment.data import *
from environment.env import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('scalar/ppo')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# vessl.init()

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.2
K_epoch = 5
T_horizon = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_pi(x)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, prob_a, done = torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
                                         torch.tensor(r_lst, dtype=torch.float).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                         torch.tensor(prob_a_lst).to(device), torch.tensor(done_lst, dtype=torch.float).to(device)

        self.data = []
        return s, a, r, s_prime, prob_a, done

    def train_net(self):
        s, a, r, s_prime, prob_a, done = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done
            delta = td_target - self.v(s)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            logit = self.pi(s)
            pi = torch.softmax(logit, dim=-1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    num_episode = 100000
    episode = 1

    score_avg = 0

    state_size = 14
    action_size = 4

    log_path = '../result/model/ppo'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/ppo'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False
    env = WeldingLine(log_dir=event_path)

    model = PPO(state_size, action_size).to(device)
    num_episode = 100000

    if load_model:
        ckpt = torch.load(log_path + "/episode9600.pt")
        model.load_state_dict(ckpt["model_state_dict"])
        model.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode = ckpt["epoch"]

    for e in range(episode, episode + num_episode + 1):
        env.e = e
        state = env.reset()
        r_epi = 0.0
        done = False

        while not done:
            for t in range(T_horizon):
                logit = model.pi(torch.from_numpy(state).float().to(device))
                prob = torch.softmax(logit, dim=-1)

                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done = env.step(action)

                model.put_data((state, action, reward, next_state, prob[action].item(), done))
                state = next_state

                r_epi += reward
                if done:
                    # print("episode: %d | reward: %.4f" % (e, r_epi))
                    # vessl.log(step=e, payload={'reward': r_epi})

                    if e % 100 == 0:
                        torch.save({"epoch": e,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": model.optimizer.state_dict()},
                                   log_path + "/episode%d.pt" % e)

                        # env.save_event_log(simulation_dir + "episode%d.csv" % e)

                    break

            model.train_net()
        print(
            "episode: %d | reward: %.4f | Setup: %.4f | Mean On-Time: %.4f | Mean Tardiness: %.4f | Mean Earliness: %.4f" % (
            e, r_epi, np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish,
            env.monitor.on_time / env.num_block, np.sum(env.monitor.tardiness) / env.num_block,
            np.sum(env.monitor.earliness) / env.num_block))

        writer.add_scalar("Reward/Reward", r_epi, e)
        # avg_loss = loss / num_update if num_update > 0 else 0
        # writer.add_scalar("Performance/Loss", avg_loss, e)
        writer.add_scalar("Performance/SetUp", np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish, e)
        writer.add_scalar("Performance/Tardiness", np.sum(env.monitor.tardiness) / env.num_block, e)
        writer.add_scalar("Performance/On-Time", env.monitor.on_time / env.num_block, e)
        writer.add_scalar("Performance/Earliness", np.sum(env.monitor.earliness) / env.num_block, e)

    writer.close()

if __name__ == '__main__':
    main()