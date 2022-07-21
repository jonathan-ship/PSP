from ppo import *
from environment.env import WeldingLine


if __name__ == "__main__":
    state_size = 14
    action_size = 4

    log_path = '../result/model/ppo'

    event_path = '../test/result/ppo'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    tardiness_list = list()
    setup_list = list()
    in_queue_list = list()

    env = WeldingLine(log_dir=event_path)
    model = PPO(state_size, action_size).to(device)
    ckpt = torch.load(log_path + "/episode9600.pt")
    model.load_state_dict(ckpt["model_state_dict"])

    for i in range(100):
        print("Episode {0}".format(i+1))
        step = 0
        done = False
        state = env.reset()
        r_epi = 0.0

        while not done:
            logit = model.pi(torch.from_numpy(state).float().to(device))
            prob = torch.softmax(logit, dim=-1)

            action = torch.argmax(prob).item()
            next_state, reward, done = env.step(action)

            model.put_data((state, action, reward, next_state, prob[action].item(), done))
            state = next_state

            r_epi += reward
            if done:
                env.monitor.save_tracer()
                break

        tardiness_list.append(sum(env.monitor.tardiness) / env.num_block)
        setup_list.append(sum(env.monitor.setup_list) / env.model["Sink"].total_finish)

        total_in_queue = 0.0
        for block in env.model["Sink"].finished.keys():
            total_in_queue += (env.model["Sink"].finished[block]["time"][1] - env.model["Sink"].finished[block]["time"][
                0]) / 1440

        in_queue_list.append(total_in_queue / env.num_block)

    print("Avg. Tardiness : {0}".format(round(np.mean(tardiness_list), 2)))
    print("Setup Ratio    : {0}".format(round(np.mean(setup_list), 2)))
    print("Avg. In_queue Time : {0}".format(np.mean(in_queue_list)))
