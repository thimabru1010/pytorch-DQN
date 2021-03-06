import math

import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder

class Trainer:
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.env = env
        self.config = config

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)

        self.outputdir = get_output_folder(self.config.output, self.config.env)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)

        self.buffer_full = False

    def train(self, pre_fr=0):
        losses1 = []
        losses2 = []
        all_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False

        state = self.env.reset()
        for fr in range(pre_fr + 1, self.config.frames + 1):
            epsilon = self.epsilon_by_frame(fr)
            action = self.agent.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.agent.buffer.add(state, action, reward, next_state, done)

            # Lembrar de ajeitar para o o codigo de dqn normal
            if self.agent.buffer.size() == self.config.max_buff and not self.buffer_full:
                self.buffer_full = True
                print("Buffer full!")
                print(self.agent.buffer.size())
                print("Start training...")

            state = next_state
            episode_reward += reward

            if self.agent.buffer.size() >= self.config.start_training:
                loss = 0
                if self.agent.buffer.size() > self.config.batch_size:
                    loss1, loss2, index = self.agent.learning(fr)
                    losses1.append(loss1)
                    losses2.append(loss2)
                    self.board_logger.scalar_summary('Loss1 per frame', fr, loss1)
                    # Talvez funcione
                    self.board_logger.scalar_summary('Loss2 per frame', fr, loss2)

                    unique, counts = np.unique(index, return_counts=True)
                    counts_dict = dict(zip(unique, counts))
                    if 0 not in counts_dict.keys():
                        counts_dict[0] = 0
                    if 1 not in counts_dict.keys():
                        counts_dict[1] = 0
                    q1_counts = counts_dict[0]
                    q2_counts = counts_dict[1]
                    self.board_logger.scalar_summary('Frequency Q1', fr, q1_counts)
                    self.board_logger.scalar_summary('Frequency Q2', fr, q2_counts)


                if fr % self.config.print_interval == 0:
                    print("frames: %5d, reward: %5f, loss1: %4f, loss2: %4f  episode: %4d  \t Q1/Q2: %d/%d" % (fr, np.mean(all_rewards[-10:]), loss1, loss2, ep_num, q1_counts, q2_counts))

                if fr % self.config.log_interval == 0:
                    self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])

                if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                    self.agent.save_checkpoint(fr, self.outputdir)

                if done:
                    state = self.env.reset()
                    all_rewards.append(episode_reward)
                    episode_reward = 0
                    ep_num += 1
                    avg_reward = float(np.mean(all_rewards[-100:]))
                    self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)

                    if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward:
                        is_win = True
                        self.agent.save_model(self.outputdir, 'best')
                        print('Ran %d episodes best 100-episodes average reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, ep_num - 100))
                        if self.config.win_break:
                            break

        if not is_win:
            print('Did not solve after %d episodes' % ep_num)
            self.agent.save_model(self.outputdir, 'last')
