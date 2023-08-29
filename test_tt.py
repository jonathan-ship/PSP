import json
import pandas as pd
from scipy import stats

from cfg import *
from agent.ppo import *
from environment.env import *
from environment.test_env import *
from create_data import *

weight = {
    60: {"ATCS": [5.535, 0.943], "COVERT": 0.4},
    80: {"ATCS": [5.953, 1.000], "COVERT": 6.6},
    100: {"ATCS": [6.122, 0.953], "COVERT": 9.0},
    160: {"ATCS": [6.852, 1.173], "COVERT": 1.4},
    240: {"ATCS": [7.482, 1.482], "COVERT": 0.9}}

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    cfg = get_cfg()
    num_block_list = [60, 80, 100, 160, 240]
    ddt_list = [0.8, 1.0, 1.2]
    pt_var_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    rl_model = ["1_720_0_1", "1_720_1_0", "1_720_7_3", "1_720_8_2"]
    rl_model += ["1e-4_1_entire_5_5", "1e-4_1_entire_6_4", "1e-4_1_entire_7_3"]
    heuristic_rule = ["SSPT", "ATCS", "MDD", "COVERT"]

    trained_model = rl_model + heuristic_rule

    simulation_dir = './output/test_1/simulation' if not cfg.use_vessl else "/output/"
    if not os.path.exists(simulation_dir):
        os.makedirs(simulation_dir)

    with open("SHI_sample.json", 'r') as f:
        block_sample = json.load(f)

    n_list = list()
    dt_list = list()
    var_list = list()
    heuristic_list = list()
    tard_p = list()
    tard_tf = list()
    setup_p = list()
    setup_tf = list()

    for num_block in num_block_list:
        sample_data = get_sample(num_block)

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
                            "{0} | {1} | {2} | Test {3} | Model = {4}".format(num_block, round(ddt, 1), round(pt_var, 1),
                                                                              test_i, model))
                        simulation_dir_rule = simulation_dir + '/{0}_{1}_{2}'.format(num_block, str(round(pt_var, 1)),
                                                                                     str(round(ddt, 1)))
                        if not os.path.exists(simulation_dir_rule):
                            os.makedirs(simulation_dir_rule)

                        if model not in ["SSPT", "ATCS", "MDD", "COVERT"]:
                            env = WeldingLine(num_block=num_block,
                                              block_sample=block_sample,
                                              rule_weight=weight[num_block],
                                              log_dir=simulation_dir_rule + '/rl_{0}_episode_{1}.csv'.format(
                                                  model, test_i),
                                              test_sample=test_data,
                                              pt_var=pt_var,
                                              ddt=ddt,
                                              is_train=False)
                            model_path = "./trained model/1_720/{0}.pt".format(model) if model[1] == "_" else "./trained model/1_entire/{0}.pt".format(model)
                            K_epoch = 1 if not cfg.use_vessl else cfg.K_epoch
                            agent = PPO(env.state_size, env.action_size, 0.0005, cfg.gamma, cfg.lmbda, cfg.eps_clip, K_epoch).to(device)
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
                                    output_tardiness[model].append(tardiness)
                                    output_setup[model].append(setup)
                                    break

                        else:  # Heuristic rule
                            tard, setup = test(num_block=num_block, block_sample=block_sample,
                                               sample_data=test_data, routing_rule=model,
                                               file_path=simulation_dir_rule + '/{0}_episode_'.format(model),
                                               ddt=ddt, pt_var=pt_var, test_num=1)
                            output_tardiness[model].append(tard)
                            output_setup[model].append(setup)

                with open(simulation_dir + "/Test Tardiness({0}, {1}, {2}).json".format(num_block,
                                                                                        str(round(pt_var,
                                                                                                  1)),
                                                                                        str(round(ddt, 1))),
                          'w') as f:
                    json.dump(output_tardiness, f)
                with open(simulation_dir + "/Test Setup({0}, {1}, {2}).json".format(num_block,
                                                                                    str(round(pt_var, 1)),
                                                                                    str(round(ddt, 1))),
                          'w') as f:
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
                        n_list.append(num_block)
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

