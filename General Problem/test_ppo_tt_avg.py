import json
from scipy import stats
import pandas as pd

from agent.ppo import *
from environment.test_env import *
from cfg import *
from create_data import *

# torch.manual_seed(42)
# random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()

    if cfg.env == "OE":
        from environment.env import *
    elif cfg.env == "EE":
        from environment.env_2 import *

    num_job_list = [100, 200, 400]

    weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
              200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
              400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    rl_model = ["1e-3_1_6_4", "5e-3_1_5_5", "5e-3_1_6_4"]
    heuristic_rule = ["SSPT", "ATCS", "MDD", "COVERT"]
    trained_model = rl_model + heuristic_rule
    # trained_model = ["SSPT", "ATCS", "MDD", "COVERT"]
    simulation_dir = './output/thesis_tt_avg/simulation' if not cfg.use_vessl else "/simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    n_list = list()
    dt_list = list()
    var_list = list()
    heuristic_list = list()
    rl_rule_list = list()
    tard_p = list()
    tard_tf = list()
    setup_p = list()
    setup_tf = list()

    for num_job in num_job_list:
        if cfg.use_vessl:
            import vessl
            vessl.init(organization="snu-eng-dgx", project="Final-General-PMSP", hp=cfg)

        sample_data = get_sample(num_job, 2000)

        output_tardiness = dict()
        output_setup = dict()

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        for test_i in sample_data.keys():
            test_data = sample_data[test_i]
            pt_var = np.random.uniform(low=0.1, high=0.5)
            ddt = np.random.uniform(low=0.8, high=1.2)
            for model in trained_model:
                if model not in output_tardiness.keys():
                    output_tardiness[model] = list()
                if model not in output_setup.keys():
                    output_setup[model] = list()

                print(
                    "{0} | {1} | {2} | Test {3} | Model = {4}".format(num_job, ddt, pt_var, test_i, model))

                if model not in ["SSPT", "ATCS", "MDD", "COVERT"]:  # 강화학습으로 학습한 모델
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
                            #     simulation_dir_rule + '/rl_{0}_episode_{1}.csv'.format(model, test_i))
                            output_tardiness[model].append(env.monitor.tardiness / env.num_job)
                            output_setup[model].append(env.monitor.setup / env.num_job)
                            break

                else:  # Heuristic rule
                    tard, setup = test(num_job=num_job, sample_data=test_data, routing_rule=model,
                                       file_path=None, ddt=ddt, pt_var=pt_var, test_num=1)
                    output_tardiness[model].append(tard)
                    output_setup[model].append(setup)

        # with open(simulation_dir + "/Test Tardiness({0}).json".format(num_job), 'w') as f:
        #     json.dump(output_tardiness, f)
        # with open(simulation_dir + "/Test Setup({0}).json".format(num_job), 'w') as f:
        #     json.dump(output_setup, f)

        for hrst in heuristic_rule:
            for rl in rl_model:
                # Heuristic이 강화학습보다 tardiness가 크다 (안 좋다)
                stat_tard, p_val_tard = stats.ttest_rel(np.array(output_tardiness[hrst]),
                                                        np.array(output_tardiness[rl]),
                                                        alternative='greater')

                # Heuristic이 강화학습보다 setup이 크다 (안 좋다)
                stat_setup, p_val_setup = stats.ttest_rel(np.array(output_setup[hrst]),
                                                          np.array(output_setup[rl]),
                                                          alternative='greater')

                heuristic_list.append(hrst)
                rl_rule_list.append(rl)
                tard_p.append(p_val_tard)
                tard_p_under = True if p_val_tard <= 0.05 else False
                tard_tf.append(tard_p_under)
                setup_p.append(p_val_setup)
                setup_p_under = True if p_val_setup <= 0.05 else False
                setup_tf.append(setup_p_under)
                n_list.append(num_job)

    temp = pd.DataFrame()
    temp["Num of Job"] = n_list
    temp["Heuristic"] = heuristic_list
    temp["RL"] = rl_rule_list
    temp["P-Value(Tard)"] = tard_p
    temp["Tardiness(<)"] = tard_tf
    temp["P-Value(Setup)"] = setup_p
    temp["Setup(<)"] = setup_tf

    temp.to_excel(simulation_dir + "/T-Test.xlsx")



