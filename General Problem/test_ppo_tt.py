import json
from scipy import stats
import pandas as pd

from agent.ppo import *
from environment.test_env import *
from cfg import *

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
    ddt_list = [0.8, 1.0, 1.2]
    pt_var_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
              200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
              400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}
    rl_model = list()
    for rw in ["8_2"]:
        for optim in ["Adam"]:
            rl_model.append("{0}_{1}".format(rw, optim))
    heuristic_rule = ["SSPT", "ATCS", "MDD", "COVERT"]
    trained_model = rl_model + heuristic_rule
    # trained_model = ["SSPT", "ATCS", "MDD", "COVERT"]
    simulation_dir = './output/test_ppo_ep1_tt_500/simulation' if not cfg.use_vessl else "/simulation"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    n_list = list()
    dt_list = list()
    var_list = list()
    heuristic_list = list()
    tard_p = list()
    tard_tf = list()
    setup_p = list()
    setup_tf = list()

    for num_job in num_job_list:
        if cfg.use_vessl:
            import vessl
            vessl.init(organization="snu-eng-dgx", project="Final-General-PMSP", hp=cfg)

        with open("./data/sample{0}.json".format(num_job), 'r') as f:
            sample_data = json.load(f)

        for ddt in ddt_list:
            for pt_var in pt_var_list:
                output_tardiness = dict()
                output_setup = dict()

                for model in trained_model:
                    output_tardiness[model] = list()
                    output_setup[model] = list()

                    for test_i in sample_data.keys():
                        test_data = sample_data[test_i]
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
                            env = PMSP(num_job=num_job,
                                       test_sample=test_data,
                                       rule_weight=weight[num_job],
                                       ddt=ddt, pt_var=pt_var, is_train=False)

                            model_path = "./tm/5e-4_local/{0}_episode-50000.pt".format(model)
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
                                    log = env.get_logs(
                                        simulation_dir_rule + '/rl_{0}_episode_{1}.csv'.format(model, test_i))
                                    output_tardiness[model].append(env.monitor.tardiness / env.num_job)
                                    output_setup[model].append(env.monitor.setup / env.num_job)
                                    break

                        else:  # Heuristic rule
                            tard, setup = test(num_job=num_job, sample_data=test_data, routing_rule=model,
                                               file_path=simulation_dir_rule + '/{0}_episode_'.format(model),
                                               ddt=ddt, pt_var=pt_var, test_num=1)
                            output_tardiness[model].append(tard)
                            output_setup[model].append(setup)

                with open(simulation_dir + "/Test Tardiness({0}, {1}, {2}).json".format(num_job, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1))), 'w') as f:
                    json.dump(output_tardiness, f)
                with open(simulation_dir + "/Test Setup({0}, {1}, {2}).json".format(num_job, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1))), 'w') as f:
                    json.dump(output_setup, f)

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
                        tard_p.append(p_val_tard)
                        tard_p_under = True if p_val_tard <= 0.05 else False
                        tard_tf.append(tard_p_under)
                        setup_p.append(p_val_setup)
                        setup_p_under = True if p_val_setup <= 0.05 else False
                        setup_tf.append(setup_p_under)
                        n_list.append(num_job)
                        dt_list.append(ddt)
                        var_list.append(pt_var)

    temp = pd.DataFrame()
    temp["Num of Job"] = n_list
    temp["DDT"] = dt_list
    temp["pt_var"] = var_list
    temp["Heuristic"] = heuristic_list
    temp["P-Value(Tard)"] = tard_p
    temp["Tardiness(<)"] = tard_tf
    temp["P-Value(Setup)"] = setup_p
    temp["Setup(<)"] = setup_tf

    temp.to_excel(simulation_dir + "/T-Test.xlsx")



