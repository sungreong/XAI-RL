import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
from rlxai.dp.img import data_transform
from rlxai.agent.agent_selection import get_agent
from rlxai.gym_wrapper import FireResetEnv, WarpFrame, ScaledFloatFrame, FrameStack
from collections import Counter
import argparse


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="BreakoutDeterministic-v4")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "DQN", "Double", "PER"])
    parser.add_argument("--save_path", type=str, default="./model")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2345)
    parser.add_argument("--print_per_episode", type=int, default=5)
    params = vars(parser.parse_known_args()[0])
    return params


def data_transform(frame):
    # breakout-v4
    # [width, height, channels] (기존)
    # [channels, height, width] (변경)
    frame = np.array(frame).transpose(2, 1, 0)
    frame = np.expand_dims(frame, 0)
    assert np.sum(np.isnan(frame)) == 0, "state error!"
    return frame.astype(np.float32)


if __name__ == "__main__":
    params = get_params()
    N_EPISODE = 1_000_000_000
    n_print = params.get("print_per_episode")
    env = gym.make(params.get("env_name"))  # Deterministic
    stack_n = 4
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ScaledFloatFrame(env)
    env = FrameStack(env, k=stack_n)

    # Deterministic
    random_seed = params.get("seed")
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    action_dim = env.action_space.n
    device = params.get("device")
    model_name = params.get("algo")
    save_path = params.get("save_path")
    agent, model_algo_path, checkpoint_path = get_agent(
        model_name=model_name,
        save_path=save_path,
        device=device,
        action_dim=action_dim,
        state_dim=stack_n,
        load_model=True,
    )
    REWARD = 0
    AVG_REWARDS = 0
    AVG_REWARD_LIST = []
    TOTAL_STEP = 0
    ACTION_MEANING = env.unwrapped.get_action_meanings()
    SAVE_ITER = 0

    for n_episode in range(1, N_EPISODE):
        state = env.reset()
        info = {"lives": 5}
        last_lives = info["lives"]
        # state_list = [data_transform(state) for _ in range(3)]
        rewards_list = []
        n_timestep = 0
        ACTION_LIST = []
        try:
            while True:
                TOTAL_STEP += 1
                n_timestep += 1
                print(
                    f"[{model_name}] TOTAL STEP : {TOTAL_STEP:06d}, EPISODE : {n_episode:05d}, TIME STEP : {n_timestep:04d} TOTAL REWARD : {int(np.sum(rewards_list)):03d}",
                    end="\r",
                )
                # state = np.concatenate(state_list, axis=1)  / 255.0
                action_dict = agent.select_action(data_transform(state), True)
                if last_lives > info["lives"]:
                    action = 1  # TODO: 에이전트가 1을 선택할 수 있게 하기?
                    last_lives = info["lives"]
                else:
                    action = int(action_dict["action"])
                n_state, reward, done, info = env.step(action)
                ACTION_LIST.append(ACTION_MEANING[action])
                # state_list.pop(0)
                # if last_lives > info["lives"]:
                #     n_state = np.zeros_like(data_transform(n_state),dtype=np.float32)
                #     state_list.append(n_state)
                # else:
                #     state_list.append(data_transform(n_state))
                # n_state = np.concatenate(state_list, axis=1)  / 255.0
                reward, done = map(lambda x: np.expand_dims(x, 0), [[reward], [done]])  # for (1, ?)
                trainsition = {
                    "state": data_transform(state),
                    "next_state": data_transform(n_state),
                    "reward": reward,
                    "done": done,
                }
                state = n_state
                trainsition.update(action_dict)
                loss_result = agent.process(transitions=[trainsition], step=TOTAL_STEP)
                # if loss_result == {}:
                #     pass
                # else:
                #     print()
                #     for k, v in loss_result.items():
                #         print(f"{k} : {v:.5f}")
                rewards_list.append(reward)
                if done:
                    break
            print("")
            print(Counter(ACTION_LIST))
            print("")
            AVG_REWARDS += np.sum(rewards_list)
            if (n_episode % n_print == 0) & (n_episode > 0):
                AVG_REWARDS /= n_print
                print()
                print("Episode : {} \t\t AVG Reward : {}".format(n_episode, AVG_REWARDS))
                AVG_REWARD_LIST.append(AVG_REWARDS)
                X_ITER = [i * n_print for i in range(len(AVG_REWARD_LIST))]
                plt.plot(X_ITER, AVG_REWARD_LIST)
                plt.title(f"SAVE MODEL Iter : {SAVE_ITER}, MAX REWARD : {REWARD}")
                plt.savefig(model_algo_path.joinpath("./avg_reward.png"))
                plt.close()
                if REWARD < AVG_REWARDS:
                    REWARD = AVG_REWARDS
                    SAVE_ITER = n_episode
                    print()
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )
                    print("saving model at : " + str(checkpoint_path))
                    agent.save(checkpoint_path)
                    print("model saved")
                    print(
                        "--------------------------------------------------------------------------------------------"
                    )
                    print()
                AVG_REWARDS = 0
        except Exception as e:
            trace = []
            tb = e.__traceback__
            while tb is not None:
                trace.append(
                    {
                        "filename": tb.tb_frame.f_code.co_filename,
                        "name": tb.tb_frame.f_code.co_name,
                        "lineno": tb.tb_lineno,
                    }
                )
                tb = tb.tb_next
            #############################################################################
            with open(model_algo_path.joinpath("error.txt"), "w") as f:
                f.write(f"ERROR....")
                f.write("\n")
                f.write(f"type : {type(e).__name__}")
                f.write("\n")
                f.write(f"message : {str(e)}")
                f.write("\n")
                f.write("\n".join([str(trace_line) for trace_line in trace]))
            raise Exception(e)
