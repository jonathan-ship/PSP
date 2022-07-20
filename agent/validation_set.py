import os, copy, json
import pandas as pd

from agent.dqn import *
from environment.env import WeldingLine


if __name__ == "__main__":
    num_episode = 9
    episode = 1

    score_avg = 0

    state_size = 14
    action_size = 4

    log_path = '../result/model/dqn'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    event_path = '../environment/validation_set'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    load_model = False

    env = WeldingLine(log_dir=event_path)
    q = Qnet(state_size, action_size)
    q_target = Qnet(state_size, action_size)
    optimizer = optim.RMSprop(q.parameters(), lr=1e-5, alpha=0.99, eps=1e-06)

    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    update_interval = 20
    score = 0

    step = 0
    moving_average = list()

    state_output = list()
    for e in range(episode, episode + num_episode + 1):
        state_list = list()
        print("#" * 20, "Episode {0}".format(e), "#" * 20)
        env.e = e
        done = False
        step = 0
        state = env.reset()

        state_list.append(copy.deepcopy(state))

        r = list()
        loss = 0
        num_update = 0

        while not done:
            epsilon = 1.0

            step += 1
            action = q.sample_action(torch.from_numpy(state).float(), epsilon)

            # 환경과 연결
            next_state, reward, done = env.step(action)
            state_list.append(copy.deepcopy(next_state))
            r.append(reward)
            memory.put((state, action, reward, next_state, done))

            # if memory.size() > 2000:
            # train(q, q_target, memory, optimizer)

            state = next_state

            if e % update_interval == 0 and e != 0:
                q_target.load_state_dict(q.state_dict())

            if done:
                print(e, sum(r), epsilon)
                moving_average.append(sum(r))

            # df = pd.DataFrame(moving_average)
            # df.to_csv("C:/Users/sohyon/PycharmProjects/UPJSP_SH/moving_average.csv")

                if e % 100 == 0:
                    torch.save({'episode': e, 'model_state_dict': q_target.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, log_path + '/episode%d.pt' % (e))
                    print('save model...')
                break
        env.monitor.save_tracer()

        idx_list = list(np.random.choice([i for i in range(len(state_list))], 5))
        for idx in idx_list:
            state_output.append(state_list[idx])

    state_output = np.array(state_output)
    np.save('validation_set', state_output)