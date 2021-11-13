import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import models
import torch
import gym, sys
import cv2
from tqdm import tqdm
from IPython.display import clear_output
from array2gif import write_gif
from rlxai import A2C

env = gym.make("Breakout-v0")

import threading


def img_crop(img_arr):
    return img_arr[55:-15, 15:-15, :]


def rgb2gray(rgb):
    image_data = cv2.cvtColor(cv2.resize(rgb, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def totensor(img_arr):
    return torch.FloatTensor(img_arr.transpose((2, 0, 1))).unsqueeze(dim=0)


def data_transform(x):
    x = img_crop(x)
    x = rgb2gray(x)
    x = totensor(x)
    return x


actor = models.resnet18(pretrained=True)
actor.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
actor.fc = nn.Linear(512, env.action_space.n)

critic = models.resnet18(pretrained=True)
critic.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
critic.fc = nn.Linear(512, 1)
A2C_MODEL = A2C(actor=actor, critic=critic, lr=1e-3, gamma=0.99, lam=0.9)


############################################################
rewards = []  #
is_terminals = []  #
values = []  #
logprobs = []  #
############################################################
max_steps_per_episode = 100_000_000
batch_size = 1000
s = env.reset()
try:
    A2C_MODEL.load_model("./model")
except Exception as e:
    print(e)
img_collection = [s.transpose((2, 0, 1))]


def save_gif(img_collection, file_path):
    write_gif(img_collection, file_path)


from copy import deepcopy

STOP_ITERATION = 100
with tqdm(total=max_steps_per_episode, file=sys.stdout) as pbar:
    for t in range(1, max_steps_per_episode):
        ###########################
        s_list = []
        for _ in range(3):
            s, r, done, info = env.step(0)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
            s_tensor = data_transform(s)
            s_list.append(s_tensor)
            if done:
                del img_collection
                img_collection = []
                s = env.reset()
        ###########################
        s_tensor = torch.cat(s_list, axis=1)
        act, v, logprob = A2C_MODEL.choose_action(s_tensor, inference=False)
        s, r, done, info = env.step(int(act))
        rewards.append(r)
        is_terminals.append(done)
        values.append(v)
        logprobs.append(logprob)
        if (batch_size == len(rewards)) | done:
            ###########################
            s_list = [data_transform(s)]
            for _ in range(2):
                s, r, done, info = env.step(0)  # ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
                s_tensor = data_transform(s)
                s_list.append(s_tensor)
            s_tensor = torch.cat(s_list, axis=1)
            ###########################
            _, v, _ = A2C_MODEL.choose_action(s_tensor, inference=False)
            actor_loss, critic_loss = A2C_MODEL.get_loss(rewards, is_terminals, values, logprobs, float(v))
            print()
            A2C_MODEL.train(actor_loss=actor_loss, critic_loss=critic_loss)
            A2C_MODEL.save_model("./model")
            ############################################################
            rewards = []  #
            is_terminals = []  #
            values = []  #
            logprobs = []  #
            ############################################################
            a_l, c_l = actor_loss.detach().numpy(), critic_loss.detach().numpy()
            if t > STOP_ITERATION:
                t1 = threading.Thread(
                    target=save_gif, args=(deepcopy(img_collection), f"./jupyter/gif/eval_{t:05d}.gif")
                )
                t1.daemon = True
                t1.start()
            if done:
                del img_collection
                img_collection = []
                print("restart...")
                s = env.reset()
                img_collection.append(s.transpose((2, 0, 1)))
        else:
            if t > STOP_ITERATION:
                img_collection.append(s.transpose((2, 0, 1)))
            a_l, c_l = 0, 0
        # pbar.set_description(f"{done} {r} {a_l:3f}, {c_l:3f}")
        pbar.update(1)
