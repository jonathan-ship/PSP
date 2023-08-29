import json
import pandas as pd

from agent.ppo import *
from environment.test_env import *
from cfg import *
from environment.env import *

# torch.manual_seed(42)
# random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()

    num_job_list = [400]
    ddt_list = [1.2]
    pt_var_list = [0.2, 0.3, 0.4, 0.5]

    weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
              200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
              400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    trained_model = ["1e-3_1_6_4", "5e-3_1_5_5"]
    trained_model += ["SSPT", "ATCS", "MDD", "COVERT"]
    # trained_model = ["SSPT", "ATCS", "MDD", "COVERT"]
    simulation_dir = './output/thesis/simulation' if not cfg.use_vessl else "/simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    for num_job in num_job_list:
        if cfg.use_vessl:
            import vessl
            vessl.init(organization="snu-eng-dgx", project="Final-General-PMSP", hp=cfg)

        with open("sample{0}.json".format(num_job), 'r') as f:
            sample_data = json.load(f)

        for ddt in ddt_list:
            for pt_var in pt_var_list:
                output_dict = dict()
                for test_i in sample_data.keys():
                    output_dict[test_i] = dict()
                    test_data = sample_data[test_i]
                    for model in trained_model:

                        torch.manual_seed(42)
                        random.seed(42)
                        np.random.seed(42)

                        print(
                            "{0} | {1} | {2} | Test {3} | Model = {4}".format(num_job, round(ddt, 1), round(pt_var, 1),
                                                                              test_i, model))
                        simulation_dir_rule = simulation_dir + '/{0}_{1}_{2}'.format(num_job, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1)))
                        if not os.path.exists(simulation_dir_rule):
                            os.makedirs(simulation_dir_rule)

                        if model not in ["SSPT", "ATCS", "MDD", "COVERT"]:  # 강화학습으로 학습한 모델
                            tard_list = list()
                            setup_list = list()

                            for i in range(100):
                                env = PMSP(num_job=num_job,
                                           test_sample=test_data,
                                           rule_weight=weight[num_job],
                                           ddt=ddt, pt_var=pt_var, is_train=False)

                                model_path = "./tm/thesis/{0}.pt".format(model)
                                agent = PPO(cfg, env.state_dim, env.action_dim).to(device)
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
                                        #     simulation_dir_rule + '/rl_{0}_episode_{1}_{2}.csv'.format(model, test_i,
                                        #                                                                i))
                                        tard_list.append(env.monitor.tardiness / env.num_job)
                                        setup_list.append(env.monitor.setup / env.num_job)
                                        break

                            output_dict[test_i][model] = {"Tardiness": np.mean(tard_list),
                                                          "Setup": np.mean(setup_list)}

                        else:  # Heuristic rule
                            tard, setup = test(num_job=num_job, sample_data=test_data, routing_rule=model,
                                               file_path=simulation_dir_rule + '/{0}_episode_'.format(model),
                                               ddt=ddt, pt_var=pt_var, test_num=100)
                            output_dict[test_i][model] = {"Tardiness": tard, "Setup": setup}

                with open(simulation_dir + "/Test Output({0}, {1}, {2}).json".format(num_job, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1))), 'w') as f:
                    json.dump(output_dict, f)

                temp = {model: {"Tardiness": 0.0, "Setup": 0.0} for model in trained_model}
                num_test = len([test_i for test_i in output_dict.keys()])
                for test_i in output_dict.keys():
                    for model in output_dict[test_i].keys():
                        temp[model]["Tardiness"] += output_dict[test_i][model]["Tardiness"] / num_test
                        temp[model]["Setup"] += output_dict[test_i][model]["Setup"] / num_test

                temp_df = pd.DataFrame(temp)
                temp_df.transpose()
                temp_df.to_excel(simulation_dir + "/Test_{0}_{1}_{2}.xlsx".format(num_job, str(round(pt_var, 1)),
                                                                                  str(round(ddt, 1))))