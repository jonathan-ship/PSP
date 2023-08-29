import json
import pandas as pd

from cfg import *
from agent.ppo import *
from environment.env import *
from environment.test_env import *

weight = {
    60: {"ATCS": [5.535, 0.943], "COVERT": 0.4},
    80: {"ATCS": [5.953, 1.000], "COVERT": 6.6},
    100: {"ATCS": [6.122, 0.953], "COVERT": 9.0},
    160: {"ATCS": [6.852, 1.173], "COVERT": 1.4},
    240: {"ATCS": [7.482, 1.482], "COVERT": 0.9}}

torch.manual_seed(42)
random.seed(42)
device = torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()
    num_block_list = [60, 80, 100, 160, 240]
    pt_var_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    # trained_model = ["1_720_0_1", "1_720_1_0", "1_720_7_3", "1_720_8_2"]
    # trained_model = ["1e-4_1_entire_0_1", "1e-4_1_entire_1_9", "1e-4_1_entire_2_8", "1e-4_1_entire_3_7", "1e-4_1_entire_4_6", "1e-4_1_entire_5_5", "1e-4_1_entire_6_4", "1e-4_1_entire_7_3", "1e-4_1_entire_8_2", "1e-4_1_entire_9_1", "1e-4_1_entire_1_0"]
    trained_model = ["SSPT", "ATCS", "MDD", "COVERT"]

    simulation_dir = './output/thesis_1/simulation' if not cfg.use_vessl else '/output'
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    with open("SHI_sample.json", 'r') as f:
        block_sample = json.load(f)

    for num_block in num_block_list:
        with open("SHI_test_sample_{0}_new.json".format(num_block), 'r') as f:
            sample_data = json.load(f)

        for pt_var in pt_var_list:
            output_dict = dict()

            for test_i in sample_data.keys():
                test_data = sample_data[test_i]
                output_dict[test_i] = dict()

                for model in trained_model:
                    torch.manual_seed(42)
                    random.seed(42)
                    np.random.seed(42)

                    print(
                        "{0} | {1} | Test {2} | Model = {3}".format(num_block, round(pt_var, 1), test_i, model))
                    simulation_dir_rule = simulation_dir + '/{0}_{1}'.format(num_block, str(round(pt_var, 1)))
                    if not os.path.exists(simulation_dir_rule):
                        os.makedirs(simulation_dir_rule)

                    if model not in ["SSPT", "ATCS", "MDD", "COVERT"]:
                        tard_list = list()
                        setup_list = list()

                        for i in range(cfg.test_num):
                            env = WeldingLine(num_block=num_block,
                                              block_sample=block_sample,
                                              rule_weight=weight[num_block],
                                              log_dir=simulation_dir_rule + '/rl_{0}_episode_{1}.csv'.format(
                                                  model, test_i),
                                              test_sample=test_data,
                                              pt_var=pt_var,
                                              is_train=False)

                            model_path = "./trained model/1_720/{0}.pt".format(model) if model[1] == "_" else "./trained model/1_entire/{0}.pt".format(model)
                            K_epoch = 1 if not cfg.use_vessl else cfg.K_epoch
                            agent = PPO(env.state_size, env.action_size, 1e-4, cfg.gamma, cfg.lmbda, cfg.eps_clip, K_epoch).to(device)
                            checkpoint = torch.load(model_path)
                            agent.load_state_dict(checkpoint["model_state_dict"])

                            state = env.reset()
                            done = False

                            while not done:
                                logit = agent.pi(torch.from_numpy(state).float().to(device))
                                prob = torch.softmax(logit, dim=-1)

                                action = torch.argmax(prob).item()
                                next_state, reward, done = env.step(action)
                                state = next_state

                                if done:
                                    # log = env.get_logs(
                                    #    simulation_dir_rule + '/rl_{0}_episode_{1}_{2}.csv'.format(model, test_i,
                                    #
                                    #                                                                i))
                                    tardiness = (np.sum(env.monitor.tardiness) / env.num_block) * 24
                                    setup = env.monitor.setup / env.model["Sink"].total_finish
                                    tard_list.append(tardiness)
                                    setup_list.append(setup)
                                    break

                        output_dict[test_i][model] = {"Tardiness": np.mean(tard_list),
                                                      "Setup": np.mean(setup_list)}

                    else:  # Heuristic rule
                        tard, setup = test(num_block=num_block, block_sample=block_sample,
                                           sample_data=test_data, routing_rule=model,
                                           file_path=simulation_dir_rule + '/{0}_episode_'.format(model),
                                           pt_var=pt_var, test_num=cfg.test_num)
                        output_dict[test_i][model] = {"Tardiness": tard, "Setup": setup}

                with open(simulation_dir + "/Test Output({0}, {1}).json".format(num_block, str(round(pt_var, 1))), 'w') as f:
                    json.dump(output_dict, f)

                temp = {model: {"Tardiness": 0.0, "Setup": 0.0} for model in trained_model}
                num_test = len([test_i for test_i in output_dict.keys()])
                for test_i in output_dict.keys():
                    for model in output_dict[test_i].keys():
                        temp[model]["Tardiness"] += output_dict[test_i][model]["Tardiness"] / num_test
                        temp[model]["Setup"] += output_dict[test_i][model]["Setup"] / num_test

                temp_df = pd.DataFrame(temp)
                temp_df.transpose()
                temp_df.to_excel(simulation_dir + "/Test_{0}_{1}.xlsx".format(num_block, str(round(pt_var, 1))))
