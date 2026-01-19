import sys
import os
import torch
import swanlab

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.append(project_root)
# -----------------------------------------------

# 现在可以优雅地引用 src 下的模块了
# 注意：文件名不要写错 (.py去掉)，类名要和代码里定义的一致
from src.environment.core_env import DismantlingEnv
from src.utils.feature_extractor import GraphFeatureExtractor
from src.models.dual_sage import DualStreamSAGE
from src.agents.ppo_agent import PPOAgent


def main():
    # ==========================================
    # 1. 定义实验配置 (Configuration)
    # 这是整个实验的"控制面板"，必须是字典格式 {Key: Value}
    # ==========================================
    config = {
        # --- 环境参数 (Environment) ---
        "n_nodes": 200,             # 节点数量
        "graph_type": "ba",         # 图类型: ba, er, ws
        "budget_ratio": 0.15,
        "lambda": 0.1,              # 奖励函数中的成本权衡系数
        "cost_alpha": 1.0,          # 成本计算系数 alpha * log(d)
        "cost_beta": 1.0,           # 成本基数
        "cost_noise": 0.1,          # 成本随机扰动范围
        "cost_model": "random",

        # --- PPO 训练参数 (Training) ---
        "lr": 3e-4,                 # 学习率
        "gamma": 0.99,              # 折扣因子
        "eps_clip": 0.2,            # PPO Clip 范围
        "k_epochs": 4,              # 每次更新循环次数
        "entropy_coef": 0.01,       # 熵正则化系数 (鼓励探索)
        "update_timestep": 2000,    # 每隔多少步更新一次
        
        # --- 模型参数 (Model) ---
        "hidden_dim": 64,           # 神经网络隐藏层维度
        
        # --- 特征提取参数 (Features) ---
        "max_degree": 100.0,        # 归一化用的预估最大度
        "max_cost": 10.0            # 归一化用的预估最大成本
    }

    # ==========================================
    # 2. 初始化 SwanLab
    # ==========================================
    swanlab.init(
        project="ESWA-Cost-Dismantling",
        experiment_name="DualSAGE-PPO-S1",
        config=config  # <--- 这里传入上面定义的字典变量
    )
    
    # ==========================================
    # 3. 初始化各模块
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # 环境
    env = DismantlingEnv(config)
    
    # 特征提取器
    feature_extractor = GraphFeatureExtractor(config, device)
    
    # 神经网络 (Brain)
    model = DualStreamSAGE(config).to(device)
    
    # 智能体 (Soul)
    agent = PPOAgent(model, config, device)
    
    # ==========================================
    # 4. 训练主循环 (Training Loop)
    # ==========================================
    MAX_EPISODES = 5000
    time_step = 0
    
    for i_episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        
        # 动态更新归一化参数，防止图变大时特征爆炸 (Optional)
        current_max_deg = max(dict(obs['graph'].degree()).values())
        current_max_cost = max(obs['node_costs'].values())
        feature_extractor.update_normalization(current_max_deg, current_max_cost)
        
        state_data = feature_extractor.convert(obs)
        ep_reward = 0
        
        while True:
            time_step += 1
            
            # --- Action Select ---
            action, log_prob, value, alpha_val = agent.select_action(state_data)
            
            # 映射回 NetworkX 的真实 ID
            nx_action_id = state_data.mapping[action]
            
            # --- Environment Step ---
            next_obs, reward, done, _, info = env.step(nx_action_id)
            
            # --- Store to Buffer ---
            # [CRITICAL FIX]
            # PyG Batching 会因为 mapping (dict with int keys) 而报错
            # 我们必须在存入 buffer 前移除它，因为训练时不需要 mapping
            
            # 1. 先转到 CPU (创建副本)
            data_for_buffer = state_data.to("cpu")
            
            # 2. 安全移除 mapping 和 num_nodes (num_nodes 会由 Batch 自动重算，存了反而报 warning)
            if hasattr(data_for_buffer, 'mapping'):
                del data_for_buffer.mapping
            if hasattr(data_for_buffer, 'num_nodes'):
                del data_for_buffer.num_nodes
            
            # 3. 存入纯净的 Tensor Data
            agent.buffer.states.append(data_for_buffer)
            
            agent.buffer.actions.append(action)
            agent.buffer.logprobs.append(log_prob)
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            agent.buffer.values.append(value)
            
            ep_reward += reward
            
            # Prepare next state
            if not done:
                state_data = feature_extractor.convert(next_obs)
            
            # --- PPO Update ---
            if time_step % config["update_timestep"] == 0:
                print(f"--- Updating PPO at step {time_step} ---")
                loss = agent.update()
                swanlab.log({"Train/Loss": loss}, step=i_episode)
                
            if done:
                break
        
        # --- Logging ---
        # 记录每个 Episode 的核心指标
        swanlab.log({
            "Train/Reward": ep_reward,
            "Train/LCC_Drop": info['lcc'], # 这里的 info 来源于 env.step 返回的最后一个 info
            "Analysis/Alpha_Mean": alpha_val.mean().item() # 监控最后一个 Alpha 的均值
        }, step=i_episode)
        
        # 打印进度
        if i_episode % 10 == 0:
            print(f"Episode {i_episode} | Reward: {ep_reward:.4f} | Nodes Left: {info['num_nodes']} | Budget Used: {1 - info['budget_ratio']:.2f}")

if __name__ == "__main__":
    main()