import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

class DismantlingEnv(gym.Env):
    """
    Cost-Sensitive Network Dismantling Environment.
    
    Paper Contribution:
    - Introduces a heterogeneous cost structure C(v).
    - Optimizes the trade-off between LCC reduction and Budget consumption.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. 配置参数加载
        self.n_nodes = config.get("n_nodes", 200)
        self.graph_type = config.get("graph_type", "ba") # ba, er, ws, file
        self.budget_ratio = config.get("budget_ratio", 0.15)
        self.initial_budget = config.get("budget", 20.0)
        
        # Reward Function Parameters
        # lambda: 权衡系数。越大则越看重省钱，越小则越看重破坏。
        self.lambda_r = config.get("lambda", 0.1) 
        
        # Cost Function Parameters
        # Cost = alpha * log(degree + 1) + beta + noise
        self.cost_info = {
            "alpha": config.get("cost_alpha", 1.0),
            "beta": config.get("cost_beta", 1.0),
            "noise": config.get("cost_noise", 0.1)
        }

        # 2. 空间定义 (虽然我们主要用 FeatureExtractor，但定义好 Space 是好习惯)
        # Action: 节点 ID (0 ~ N-1)
        self.action_space = spaces.Discrete(self.n_nodes)
        # Obs: 我们返回的是字典，由 FeatureExtractor 处理
        self.observation_space = spaces.Dict() 

        # 运行时状态占位符
        self.base_graph = None     # 初始图 (备份用)
        self.current_graph = None  # 当前残余图
        self.node_costs = {}       # 节点成本字典
        self.current_budget = 0.0  # 剩余预算
        self.removed_nodes = set() # 已删除节点集合
        
        # 统计信息
        self.initial_lcc = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # 1. 生成图
        self.base_graph = self._generate_graph()
        self.current_graph = self.base_graph.copy()
        
        # 2. 生成成本 (Assign Costs)
        self._assign_costs()
        
        # [CHANGE] 3. 基于总成本计算初始预算
        # Total Cost of the world
        total_system_cost = sum(self.node_costs.values())
        
        # Budget = Total * Ratio
        self.initial_budget = total_system_cost * self.budget_ratio
        self.current_budget = self.initial_budget
        
        # 3. 初始化状态
        self.removed_nodes = set()
        self.initial_lcc = len(max(nx.connected_components(self.base_graph), key=len))
        self.prev_lcc = self.initial_lcc
        
        # 返回 FeatureExtractor 需要的原始字典
        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        核心交互逻辑。
        Action 是 NetworkX 的节点 ID（也就是 FeatureExtractor 映射回来的 ID）。
        """
        # --- 1. 逻辑检查 (Sanity Check) ---
        err_msg = ""
        # 检查是否重复删除
        if action in self.removed_nodes:
            # 在 PPO Masking 机制下，理论上不应发生，但作为 Environment 必须兜底
            # 给予极大惩罚并强制结束，防止死循环
            return self._get_obs(), -1.0, True, False, {"error": "Repeat Action"}
        
        # 检查预算是否足够
        cost = self.node_costs[action]
        if cost > self.current_budget:
            # 同样，Masking 应该防止这种情况
            return self._get_obs(), -1.0, True, False, {"error": "Over Budget"}

        # --- 2. 执行物理动作 (Physics Execution) ---
        self.current_graph.remove_node(action)
        self.removed_nodes.add(action)
        self.current_budget -= cost
        
        # --- 3. 计算奖励 (Mathematical Reward Definition) ---
        # 这是一个计算密集型操作，大图需要优化，小图直接跑
        if self.current_graph.number_of_nodes() > 0:
            current_lcc = len(max(nx.connected_components(self.current_graph), key=len))
        else:
            current_lcc = 0
            
        # Term A: Normalized Damage (破坏收益) [0, 1]
        # (LCC_t - LCC_{t+1}) / Initial_N
        damage = (self.prev_lcc - current_lcc) / self.n_nodes
        
        # Term B: Normalized Cost (经济惩罚) [0, 1]
        # Cost / Max_Possible_Cost (近似) 或 Cost / Initial_Budget
        # 这里为了数值稳定性，建议归一化到 Initial Budget
        cost_penalty = cost / self.initial_budget
        
        # Total Reward
        # 这里的 lambda 是静态的。如果你想做动态 lambda，可以引入 remaining_budget 因子
        reward = damage - self.lambda_r * cost_penalty
        
        # --- 4. 更新状态与终止判定 (Termination) ---
        self.prev_lcc = current_lcc
        
        # 判定条件 1: 网络彻底瓦解 (LCC < 10% 初始大小)
        is_shattered = current_lcc <= 0.10 * self.n_nodes
        
        # 判定条件 2: 破产 (剩余预算 < 最小剩余节点成本)
        # 寻找当前图中剩下的最便宜的节点
        remaining_nodes = list(self.current_graph.nodes())
        if remaining_nodes:
            min_remaining_cost = min([self.node_costs[n] for n in remaining_nodes])
            is_bankrupt = self.current_budget < min_remaining_cost
        else:
            is_bankrupt = True # 图空了
            
        done = is_shattered or is_bankrupt
        
        return self._get_obs(), reward, done, False, self._get_info()

    def _generate_graph(self):
        """支持多种拓扑结构，增强论文泛化性实验的说服力"""
        n = self.n_nodes
        if self.graph_type == "ba":
            # Barabasi-Albert (Scale-free)
            # m=3 意味着每个新节点连接3个旧节点
            return nx.barabasi_albert_graph(n, 3)
        elif self.graph_type == "er":
            # Erdos-Renyi (Random)
            return nx.erdos_renyi_graph(n, 0.05)
        elif self.graph_type == "ws":
            # Watts-Strogatz (Small-world)
            return nx.watts_strogatz_graph(n, k=6, p=0.1)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def _assign_costs(self):
        """
        Assign costs to nodes based on the configuration.
        Supports:
        1. "degree": Log-linear relationship with degree (Paper Default).
        2. "constant": All costs = 1.0 (Degraded Experiment / Baseline).
        """
        # 从 config 中读取模式，默认为 degree
        mode = self.config.get("cost_model", "degree")
        
        self.node_costs = {}
        
        if mode == "constant":
            # ==========================
            # 退化实验: 所有节点价格为 1
            # ==========================
            for n in self.base_graph.nodes():
                self.node_costs[n] = 1.0
                
        elif mode == "degree":
            # ==========================
            # 论文主实验: 成本与度数成正比
            # ==========================
            degrees = dict(self.base_graph.degree())
            for n in self.base_graph.nodes():
                deg = degrees[n]
                c = self.cost_info["alpha"] * np.log(deg + 1) + self.cost_info["beta"]
                c += np.random.uniform(0, self.cost_info["noise"])
                self.node_costs[n] = max(0.1, c)
        
        elif mode == "random":
            # ==========================
            # 随机实验: 成本完全随机，与拓扑无关
            # ==========================
            for n in self.base_graph.nodes():
                self.node_costs[n] = np.random.uniform(1.0, 5.0)
                
        else:
            raise ValueError(f"Unknown cost_model: {mode}")

    def _get_info(self):
        """
        提供给 Train Loop 的辅助统计信息。
        train.py 中的日志记录依赖于这些键值。
        """
        # 防止 initial_budget 极小导致的除零错误
        safe_budget = self.initial_budget if self.initial_budget > 1e-5 else 1.0
        
        return {
            "lcc": self.prev_lcc,
            "budget_ratio": self.current_budget / safe_budget,
            "num_nodes": self.current_graph.number_of_nodes()
        }

    def _get_obs(self):
        """
        返回给 FeatureExtractor 的原始数据字典。
        必须包含所有 FeatureExtractor 需要的键。
        """
        return {
            "graph": self.current_graph,
            "node_costs": self.node_costs,
            "budget": self.current_budget,
            "removed_nodes": self.removed_nodes,
            
            # [CRITICAL FIX] 必须加上这一行！
            # 否则 FeatureExtractor 在计算 budget_norm = budget / initial_budget 时会报错
            "initial_budget": self.initial_budget 
        }