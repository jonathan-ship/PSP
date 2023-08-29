import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="OE", help="OE | EE")
    parser.add_argument("--use_vessl", type=bool, default=False, help="whether using vessl or not")
    parser.add_argument("--load_model", type=bool, default=False, help="load the trained model")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--n_episode", type=int, default=50000, help="Number of episodes to train")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x frames")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x frames")
    parser.add_argument("--num_job", type=int, default=100, help="the Number of jobs")
    parser.add_argument("--num_machine", type=int, default=5, help="the Number of machine")
    parser.add_argument("--weight_tard", type=float, default=0.5, help="Reward weight of tardiness")
    parser.add_argument("--weight_setup", type=float, default=0.5, help="Reward weight of setup")

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor gamma")
    parser.add_argument("--lmbda", type=float, default=0.95, help="")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="")
    parser.add_argument("--K_epoch", type=int, default=5, help="")
    parser.add_argument("--T_horizon", type=int, default=50, help="")
    parser.add_argument("--optim", type=str, default="Adam", help="optimizer, Adam | AdaHessian")

    args = parser.parse_args()

    return args