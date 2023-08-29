import argparse


def get_cfg():

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_vessl", type=bool, default=False)
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--max_episode", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num_block", type=int, default=80, help="number of blocks")
    # parser.add_argument("--num_agents", type=int, default=1, help="number of agents")

    parser.add_argument("--weight_tard", type=float, default=0.5, help="Reward weight of tardiness")
    parser.add_argument("--weight_setup", type=float, default=0.5, help="Reward weight of setup")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=5, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=50, help="running horizon")
    # parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--tt_number", type=int, default=300, help="T-Test number")
    parser.add_argument("--test_num", type=int, default=100)

    # parser.add_argument("--w_delay", type=float, default=1.0, help="weight for minimizing delays")
    # parser.add_argument("--w_move", type=float, default=0.5, help="weight for minimizing the number of ship movements")
    # parser.add_argument("--w_priority", type=float, default=0.5, help="weight for maximizing the efficiency")

    return parser.parse_args()