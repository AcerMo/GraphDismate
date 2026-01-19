import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as pyg_softmax

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = [] # 用于计算 Advantage
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

class PPOAgent(nn.Module):
    """
    PPO Agent tailored for Variable Action Spaces in Graphs.
    
    Key Innovation:
    - Uses graph-aware softmax (pyg_softmax) to handle dynamic action spaces.
    - Implements Strict Masking in both Sampling and Update phases.
    """
    def __init__(self, model, config, device):
        super(PPOAgent, self).__init__()
        self.device = device
        
        # Hyperparameters
        self.lr = config.get("lr", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.eps_clip = config.get("eps_clip", 0.2)
        self.K_epochs = config.get("k_epochs", 4)
        self.entropy_coef = config.get("entropy_coef", 0.01) # 鼓励探索
        
        self.policy = model.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        """
        在与环境交互时调用 (Rollout Phase)。
        """
        with torch.no_grad():
            self.policy.eval()
            # 1. Forward Pass
            # logits: [N, 1]
            # value: [1, 1] (Assuming batch_size=1 during rollout)
            logits, value, alpha = self.policy(state)
            
            # 2. Reshape & Masking
            logits = logits.view(-1) # [N]
            mask = state.mask.view(-1) # [N]
            
            # Strict Masking: 设置非法动作为负无穷
            # 注意：-1e9 足矣，不要用 -inf，否则计算 log_prob 可能出现 NaN
            logits = logits.masked_fill(~mask, -1e9)
            
            # 3. Softmax & Sampling
            # 因为 Rollout 是一次处理一张图，标准 softmax 即可
            probs = torch.softmax(logits, dim=0)
            
            # 再次确认：防止 numerical instability 导致非法动作有非零概率
            # 强制将 mask 部分置 0 并重新归一化
            probs = probs * mask.float()
            probs = probs / (probs.sum() + 1e-8)
            
            dist = Categorical(probs)
            action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item(), alpha

    def update(self):
        """
        PPO Update Phase (The complex part).
        """
        # 1. 准备数据
        # 将 list of Data 转换为 PyG Batch
        # 这会把所有小图拼成一个超大图 (Disjoint Union)
        old_states_batch = Batch.from_data_list(self.buffer.states).to(self.device)
        
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.buffer.values, dtype=torch.float32).to(self.device)
        
        # 2. Monte Carlo Estimate of Returns (Simple version)
        # 进阶版建议使用 GAE，这里为了代码清晰展示 Reward-to-Go
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # Normalize returns (Critical for stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Advantages
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 3. PPO Optimization Loop
        self.policy.train()
        for _ in range(self.K_epochs):
            # 重新评估当前策略下的 LogProb, Value, Entropy
            # 这里的 logits 是 [Total_Nodes_In_Batch, 1]
            logits, values, _ = self.policy(old_states_batch)
            logits = logits.view(-1)
            values = values.view(-1)
            
            # === Graph-Aware Softmax Logic ===
            # 我们需要对 batch 中的每个图分别做 softmax
            # old_states_batch.batch 是一个索引向量 [0, 0, ..., 1, 1, ..., K, K]
            # PyG 提供了 scatter_softmax (alias as pyg_softmax)
            
            # Masking (Batch level)
            mask = old_states_batch.mask.view(-1)
            logits = logits.masked_fill(~mask, -1e9)
            
            # Graph-level Softmax
            # 这里的 batch index 告诉 softmax 哪些节点属于同一个图
            probs = pyg_softmax(logits, old_states_batch.batch)
            
            # Numeric Stability Fix
            probs = probs * mask.float()
            # 重新归一化每个图的概率和 (防止 mask 掉所有节点导致的除0，虽然理论不应发生)
            # 使用 scatter_add 计算每个图的 sum，然后除
            # 为简单起见，只要 eps 处理好通常没问题
            
            # === Action Selection Logic ===
            # 我们需要获取“采取的动作”对应的概率
            # 难点：old_actions 存储的是每个图内部的相对索引 (0 ~ N_i-1)
            # 但 logits 是打平的。我们需要计算 global index。
            
            # 计算每个图的起始偏移量 (ptr)
            ptr = old_states_batch.ptr[:-1] # [0, N_0, N_0+N_1, ...]
            # global_action_indices = local_index + offset
            global_action_indices = old_actions + ptr
            
            # 获取对应动作的概率
            action_probs = probs[global_action_indices]
            dist_entropy = -(probs * (probs + 1e-9).log()).sum() # Approx global entropy
            # 更准确的熵应该是 sum over graphs then mean，这里简化处理
            
            new_logprobs = torch.log(action_probs + 1e-9)
            
            # 4. Loss Calculation
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.mse_loss(values, returns)
            
            loss = loss_actor + 0.5 * loss_critic - self.entropy_coef * dist_entropy
            
            # 5. Backprop
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping (Essential for GNN stability)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        return loss.item()