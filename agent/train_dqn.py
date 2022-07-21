import copy, os

from agent.dqn import *
from environment.env import WeldingLine

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('scalar/dqn')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    num_episode = 100000
    episode = 1

    score_avg = 0

    state_size = 14
    action_size = 4

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/dqn'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False

    env = WeldingLine(log_dir=event_path)
    q = Qnet(state_size, action_size).to(device)
    q_target = Qnet(state_size, action_size).to(device)
    optimizer = optim.RMSprop(q.parameters(), lr=1e-5, alpha=0.99, eps=1e-06)

    if load_model:
        ckpt = torch.load(log_path + "/episode4500.pt")
        q.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        episode = ckpt["episode"]

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    update_interval = 20
    score = 0

    step = 0
    moving_average = list()

    for e in range(episode, episode + num_episode + 1):
        env.e = e
        done = False
        step = 0
        state = env.reset()
        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = max(0.01, 0.1-0.01*(e/200))

            step += 1
            action = q.sample_action(torch.from_numpy(state).float().to(device), epsilon)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            r.append(reward)
            memory.put((state, action, reward, next_state, done))

            if memory.size() > 2000:
                loss += train(q, q_target, memory, optimizer)
                num_update += 1

            state = next_state

            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                print(e, sum(r), epsilon)
                moving_average.append(sum(r))

                # df = pd.DataFrame(moving_average)
                # df.to_csv("C:/Users/sohyon/PycharmProjects/UPJSP_SH/moving_average.csv")

                if e % 100 == 0:
                    torch.save({'episode': e,
                                'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               log_path + '/episode%d.pt' % (e))

                    print('save model...')

                break

        q.eval()
        validation_state = np.load('validation_set.npy')
        out = q(torch.from_numpy(validation_state).float().to(device)).cpu().detach().numpy()
        out = np.max(out, axis=1)
        avg_q = np.mean(out)

        total_in_queue = 0.0
        for block in env.model["Sink"].finished.keys():
            total_in_queue += (env.model["Sink"].finished[block]["time"][1] - env.model["Sink"].finished[block]["time"][
                0]) / 1440

        writer.add_scalar("Reward/Reward", sum(r), e)

        writer.add_scalar("Performance/Q-Value", avg_q, e)
        writer.add_scalar("Performance/SetUp", np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish, e)
        writer.add_scalar("Performance/Tardiness", np.sum(env.monitor.tardiness) / env.num_block, e)
        writer.add_scalar("Performance/In-Queue-Time", total_in_queue / env.num_block, e)
    writer.close()
