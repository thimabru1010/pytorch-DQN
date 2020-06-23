import argparse
import os
import random
import torch
from torch.optim import Adam
from tester_clipped import Tester
from buffer import ReplayBuffer
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from core.util import get_class_attr_val
from model import CnnDQN
from trainer_clipped import Trainer

class CnnDDQNAgent:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        # Instance model 1, target network 1 and the optimizer
        #print(self.config.state_shape)
        self.model1 = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model1 = CnnDQN(self.config.state_shape, self.config.action_dim)

        self.target_model1.load_state_dict(self.model1.state_dict())

        self.model_optim1 = Adam(self.model1.parameters(), lr=self.config.learning_rate)

        # Instance model 2, target network 2 and the optimizer
        self.model2 = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model2 = CnnDQN(self.config.state_shape, self.config.action_dim)

        self.target_model2.load_state_dict(self.model2.state_dict())

        self.model_optim2 = Adam(self.model2.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        # Network 1 makes the action
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model1.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()

        # Calculates Q-value for Network 1
        q_values1 = self.model1(s0).cuda()
        next_q_values1 = self.model1(s1).cuda()
        next_q_state_values1 = self.target_model1(s1).cuda()

        q_value1 = q_values1.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value1 = next_q_state_values1.gather(1, next_q_values1.max(1)[1].unsqueeze(1)).squeeze(1)

        # Calculates Q-value for Network 2
        q_values2 = self.model2(s0).cuda()
        #next_q_values2 = self.model2(s1).cuda()
        next_q_state_values2 = self.target_model2(s1).cuda()

        q_value2 = q_values2.gather(1, a.unsqueeze(1)).squeeze(1)
        # Remeber: res2 = target_network2(sP, argmax(network1(s,a) ) )
        next_q_value2 = next_q_state_values2.gather(1, next_q_values1.max(1)[1].unsqueeze(1)).squeeze(1)

        # print('='*60)
        # print('Targets:')
        # print(next_q_value1.size())
        # print(next_q_value2.size())
        # next_q_values = torch.cat((next_q_value1.unsqueeze(1), next_q_value2.unsqueeze(1)), dim=-1)
        # print(next_q_values.shape)
        # index = torch.argmin(next_q_values, dim=-1)
        # print(index.size())
        # print(next_q_values[:, index].shape)
        next_q_value = torch.min(next_q_value1, next_q_value2)
        #print(next_q_value)

        # Calculate the target
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        #expected_q_value = r + self.config.gamma * next_q_values[index] * (1 - done)
        # Notice that detach the expected_q_value
        expected_q_value = expected_q_value.detach()
        # Calculate the loss for network1
        loss1 = (q_value1 - expected_q_value).pow(2).mean()
        # Calculate the loss for network2
        # Train respect to the value of network 2 (e.g. q_value2)
        loss2 = (q_value2 - expected_q_value).pow(2).mean()

        self.model_optim1.zero_grad()
        loss1.backward()
        self.model_optim1.step()

        self.model_optim2.zero_grad()
        loss2.backward()
        self.model_optim2.step()

        if fr % self.config.update_tar_interval == 0:
            # Update target network 1
            self.target_model1.load_state_dict(self.model1.state_dict())
            # Update target network 2
            self.target_model2.load_state_dict(self.model2.state_dict())

        index=0
        return loss1.item(), loss2.item(), index

    def cuda(self):
        self.model1.cuda()
        self.target_model1.cuda()

        self.model2.cuda()
        self.target_model2.cuda()

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model1.load_state_dict(model['model'])
        else:
            self.model1.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model1.state_dict(), '%s/model_%s.pkl' % (output, name))

        # torch.save(self.model2.state_dict(), '%s/model2/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model1.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model1.load_state_dict(checkpoint['model'])
        self.target_model1.load_state_dict(checkpoint['model'])
        return fr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--clipped', type=int, default=0, help='Set 1 to train clipped ddqn model. Otherwise 0')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    args = parser.parse_args()
    # atari_ddqn.py --train --env PongNoFrameskip-v4

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 30000
    # Number of steps. Each step is a frame
    config.frames = 2000000
    config.start_training = 2000
    config.use_cuda = True
    config.learning_rate = 1e-4
    #config.max_buff = 100000
    config.max_buff = 50000
    config.update_tar_interval = 1000
    config.batch_size = 32
    # Deixar esses dois abaixo iguais
    config.print_interval = 10000
    config.log_interval = 10000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    #config.checkpoint_interval = 5000
    config.win_reward = 15  # PongNoFrameskip-v4
    config.win_break = True

    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    agent = CnnDDQNAgent(config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test(debug=True)

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
