import os
import torch
import time
from torch import nn
import numpy as np
import copy
from torch import Tensor
from collections import deque, namedtuple

from bidexhands.algorithms.rl.ppo import PPO
from torch.distributions import MultivariateNormal

ReplayBuffer = namedtuple("ReplayBuffer", ["obs", "state", "act", "rew", "done", 
                                           "log_prob", "obs_next", "adv", "vs",
                                           "value", "mu", "sigma", "log_last"])
StorageRecord = namedtuple("StorageRecord", ["storage", "last_obs"])

def weight_discount_cum(adv, done, weight, gamma):
    adv = copy.deepcopy(adv)
    for i in reversed(range(len(adv)-1)):
        adv[i] = adv[i] + weight[i + 1] * gamma * adv[i + 1] * (1 - done[i])
    return adv

def v_trace(log_rhos:Tensor,
            gamma: float, 
            rewards: Tensor,
            values: Tensor, 
            values_next: Tensor,
            done: Tensor,
            clip_rho_threshold: float=1.0,
            clip_cs_threshold: float=1.0,
            lam: float=0.95
            ):
    """Compute v-trace targets for PPO.
    
    Espeholt, L. et al. IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures. in Proceedings of the 35th International Conference on Machine Learning 1407–1416 (PMLR, 2018).
    
    Args:
        log_rhos: A float32 tensor of shape [T, B] representing the log importance sampling weights.
        gamma: A float32 scalar representing the discounting factor.
        rewards: A float32 tensor of shape [T, B] representing the rewards.
        values: A float32 tensor of shape [T, B] representing the value function estimates wrt. the
            target policy.
        bootstrap_value: A float32 of shape [B] representing the bootstrap value at time T.
        clip_rho_threshold: A float32 scalar representing clipping threshold for importance weights.
        clip_cs_threshold: A float32 scalar representing clipping threshold for cs values.
    Returns: A float32 tensor of shape [T, B].
    """
    # compute v_next
    # compute delta V
    rho = torch.exp(log_rhos)
    clipped_rho = torch.clamp(rho, max=clip_rho_threshold)
    delta_V = clipped_rho * (rewards + gamma * values_next * (1 - done) - values)
    # compute v trace
    cs = torch.clamp(rho, max=clip_cs_threshold) * lam
    # compute acc
    acc = torch.zeros_like(values[-1])
    result = []
    for t in range(values.shape[0] - 1, -1, -1):
        acc = delta_V[t] + gamma * cs[t] * acc * (1 - done[t])
        result.append(acc)
    result.reverse()
    vs_minus_v_xs = torch.stack(result)
    vs = torch.add(vs_minus_v_xs, values)
    return vs

class ATPPO(PPO):
    def __init__(self, 
                 vec_env, 
                 cfg_train, 
                 device='cpu', 
                 sampler='sequential', 
                 log_dir='run', 
                 is_testing=False, 
                 print_log=True, 
                 apply_reset=False, 
                 asymmetric=False,
                 history_len=1,
                 at_lam=0.86):
        super().__init__(vec_env, cfg_train, device, sampler, log_dir, 
                         is_testing, print_log, apply_reset, asymmetric)
        self.replay_queue = deque(maxlen=history_len)
        self.replay_buffer = None
        self.at_lam = at_lam
        self.history_len = history_len
        self.beta = 1
        self.target_kl = 0.1
    
    def process_buffer(self):
        obs_list = []
        obs_next_list = []
        state_list = []
        act_list = []
        rew_list = []
        done_list = []
        log_prob_list = []
        value_list = []
        mu_list = []
        sigma_list = []
        
        for storage, last_obs in self.replay_queue:
            obs_list.append(storage.observations)
            obs_next_list.append(torch.cat((storage.observations[1:], last_obs.unsqueeze(0)), dim=0))
            state_list.append(storage.states)
            act_list.append(storage.actions)
            rew_list.append(storage.rewards)
            done_list.append(storage.dones)
            log_prob_list.append(storage.actions_log_prob)
            value_list.append(storage.values)
            mu_list.append(storage.mu)
            sigma_list.append(storage.sigma)
        
        obs = torch.cat(obs_list, dim=0)
        obs_next = torch.cat(obs_next_list, dim=0)
        state = torch.cat(state_list, dim=0)
        state_next = state
        act = torch.cat(act_list, dim=0)
        rew = torch.cat(rew_list, dim=0)
        done = torch.cat(done_list, dim=0)
        log_prob = torch.cat(log_prob_list, dim=0)
        value = torch.cat(value_list, dim=0)
        mu = torch.cat(mu_list, dim=0)
        sigma = torch.cat(sigma_list, dim=0)
        
        vs, adv = self.compute_atrace(obs, state, act, rew, obs_next, 
                                        state_next, done, log_prob)
        with torch.no_grad():
            log_last,_,_,_,_ = self.actor_critic.evaluate(obs, state, act)
            log_last = log_last.unsqueeze(-1)
        self.replay_buffer = ReplayBuffer(obs, state, act, rew, done, log_prob, obs_next, adv, vs, value, mu, sigma, log_last)
        
    
    def compute_atrace(self, obs, state, act, rew, obs_next, state_next, done, log_prob):
        _, _, val, mu, log_std = self.actor_critic.act(obs, state)
        log_std = log_std[0]
        covariance = torch.diag(log_std.exp() * log_std.exp())
        distribution = MultivariateNormal(mu, scale_tril=covariance)
        log_p = distribution.log_prob(act)
        _, _, val_next, _, _ = self.actor_critic.act(obs_next, state_next)
        log_p = log_p.unsqueeze(-1)
        log_rho = log_p - log_prob
        vs = v_trace(log_rho, self.gamma, rew, val, val_next, done, lam=.95)
        
        # compute adv_trace
        adv = rew + self.gamma * val_next * (1 - done) - val
        weight = torch.exp(log_rho)
        weight = torch.clamp(weight, max=1.0)
        adv = weight_discount_cum(adv, done, weight, self.gamma * self.at_lam)
        
        return vs, adv
    
    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []

                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _ = self.actor_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.replay_queue.append(StorageRecord(self.storage, current_obs))
                self.process_buffer()
                # self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % log_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(num_learning_iterations)))
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches, len(self.replay_queue))
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                obs_batch = self.replay_buffer.obs.view(-1, *self.replay_buffer.obs.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.replay_buffer.state.view(-1, *self.replay_buffer.state.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.replay_buffer.act.view(-1, self.replay_buffer.act.size(-1))[indices]
                target_values_batch = self.replay_buffer.value.view(-1, 1)[indices]
                returns_batch = self.replay_buffer.vs.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.replay_buffer.log_prob.view(-1, 1)[indices]
                last_actions_log_prob_batch = self.replay_buffer.log_last.view(-1, 1)[indices]
                advantages_batch = self.replay_buffer.adv.view(-1, 1)[indices]
                old_mu_batch = self.replay_buffer.mu.view(-1, self.replay_buffer.act.size(-1))[indices]
                old_sigma_batch = self.replay_buffer.sigma.view(-1, self.replay_buffer.act.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch,
                                                                                                                       states_batch,
                                                                                                                       actions_batch)

                # KL
                # if self.desired_kl != None and self.schedule == 'adaptive':

                #     kl = torch.sum(
                #         sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                #     kl_mean = torch.mean(kl)

                #     if kl_mean > self.desired_kl * 2.0:
                #         self.step_size = max(1e-5, self.step_size / 1.5)
                #     elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                #         self.step_size = min(1e-2, self.step_size * 1.5)

                #     for param_group in self.optimizer.param_groups:
                #         param_group['lr'] = self.step_size
                
                last_ratio = actions_log_prob_batch - torch.squeeze(last_actions_log_prob_batch)
                last_ratio = torch.clip(last_ratio, -10, 10)
                approx_kl = torch.mean(torch.exp(last_ratio) - 1 - last_ratio)
                if self.target_kl is not None:
                    if approx_kl > 1.5 * self.target_kl:
                        self.beta = 2 * self.beta
                    elif approx_kl < self.target_kl / 1.5:
                        self.beta = self.beta / 2
                self.beta = np.clip(self.beta, 0.1, 10)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.beta * approx_kl

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss