from agent.ppo import *
from environment.env2 import WeldingLine

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('scalar/ppo2')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

if __name__ == "__main__":
    num_episode = 100000
    episode = 1

    score_avg = 0

    state_size = 14
    action_size = 4

    log_path = '../result/model/ppo2'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/result/ppo2'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = True
    env = WeldingLine(log_dir=event_path)

    model = PPO(state_size, action_size).to(device)
    num_episode = 100000

    if load_model:
        ckpt = torch.load(log_path + "/episode17000.pt")
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
        total_in_queue = 0.0
        for block in env.model["Sink"].finished.keys():
            total_in_queue += (env.model["Sink"].finished[block]["time"][1] - env.model["Sink"].finished[block]["time"][
                0]) / 1440

        print(
            "episode: %d | reward: %.4f | Setup: %.4f | Tardiness %.4f | In-Queue Time %.4f" % (
            e, r_epi, np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish,
             np.sum(env.monitor.tardiness) / env.num_block, total_in_queue / env.num_block))


        writer.add_scalar("Reward/Reward", r_epi, e)
        # avg_loss = loss / num_update if num_update > 0 else 0
        # writer.add_scalar("Performance/Loss", avg_loss, e)
        writer.add_scalar("Performance/SetUp", np.sum(env.monitor.setup_list) / env.model["Sink"].total_finish, e)
        writer.add_scalar("Performance/Tardiness", np.sum(env.monitor.tardiness) / env.num_block, e)
        #writer.add_scalar("Performance/On-Time", env.monitor.on_time / env.num_block, e)
        # writer.add_scalar("Performance/Earliness", np.sum(env.monitor.earliness) / env.num_block, e)
        writer.add_scalar("Performance/In-Queue-Time", total_in_queue / env.num_block, e)

    writer.close()