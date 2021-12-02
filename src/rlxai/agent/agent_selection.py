from rlxai.agent.per import PER
from rlxai.agent.dqn import DQN
from rlxai.agent.double import Double
from rlxai.agent.ppo_v2 import PPO
from pathlib import Path


def get_agent(model_name, save_path, device, state_dim, action_dim, load_model=True):
    if model_name == "Double":
        agent = Double(
            state_size=state_dim,
            action_size=action_dim,
            device=device,
            epsilon_init=0.7,
            buffer_size=100_000,
            start_train_step=50000,
            target_update_period=10_000,
            batch_size=256,
            lr=1e-4,
        )
    elif model_name == "PER":
        agent = PER(
            state_size=state_dim,
            action_size=action_dim,
            epsilon_init=0.7,
            device=device,
            buffer_size=100_000,
            start_train_step=50000,
            target_update_period=10_000,
            batch_size=256,
            lr=1e-4,
            alpha=0.6,
            beta=0.4,
            learn_period=16,
            uniform_sample_prob=1e-3,
        )
    elif model_name == "PPO":
        agent = PPO(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=5,
            batch_size=128,
            eps_clip=0.2,
            device=device,
            n_step=5000,
        )
    elif model_name == "DQN":
        agent = DQN(
            state_size=state_dim,
            action_size=action_dim,
            epsilon_init=0.7,
            device=device,
            buffer_size=100_000,
            start_train_step=50000,
            target_update_period=10_000,
            batch_size=256,
            lr=1e-4,
        )

    model_path = Path(save_path)
    model_algo_path = model_path.joinpath(model_name)
    model_algo_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = model_algo_path.joinpath("./model.pt")
    try:
        print("load")
        if load_model:
            agent.load(checkpoint_path)
    except Exception as e:
        print(e)
        pass
    return agent, model_algo_path, checkpoint_path
