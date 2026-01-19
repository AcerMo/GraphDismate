import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

class GraphFeatureExtractor:
    """
    Bridge between NetworkX (Environment) and PyG (Model).
    
    Attributes:
        normalization_consts (dict): Stores initial max_degree and max_cost for stable scaling.
        device (torch.device): Computation device.
    """
    def __init__(self, config, device):
        self.device = device
        # 从配置中读取归一化常数（如果在训练开始前已知）
        # 或者设计为自适应模式，在 reset() 时获取
        self.norm_degree = config.get("max_degree", 100.0) 
        self.norm_cost = config.get("max_cost", 10.0)
        self.initial_budget = config.get("budget", 1.0)

    def update_normalization(self, max_degree, max_cost):
        """
        在每个 Episode 开始时调用，确保归一化基准是基于初始图的。
        这消除了 Input Distribution Shift 问题。
        """
        self.norm_degree = max(max_degree, 1.0)
        self.norm_cost = max(max_cost, 0.1)

    def convert(self, obs):
        """
        Core method: Converts raw environment observation to PyG Data object.
        
        Args:
            obs (dict): {
                'graph': nx.Graph (Residual graph),
                'node_costs': dict {node_id: cost},
                'budget': float (Remaining budget),
                'removed_nodes': set
            }
            
        Returns:
            data (torch_geometric.data.Data): Ready for GNN.
                - x: [N_current, 2] (Normalized Degree, Normalized Cost)
                - edge_index: [2, E_current]
                - mask: [N_current] (Action feasibility mask)
                - mapping: {pyg_idx: nx_id}
        """
        G = obs['graph']
        budget = obs['budget']
        node_costs = obs['node_costs']
        
        episode_initial_budget = obs['initial_budget']

        # 1. 建立动态映射 (Dynamic Mapping)
        # PyG 需要连续的 0 ~ N-1 索引
        # 我们只处理残余图中的节点 (Inductive setting)
        pyg_idx_to_nx_id = {i: n for i, n in enumerate(G.nodes())}
        nx_id_to_pyg_idx = {n: i for i, n in enumerate(G.nodes())}
        
        num_nodes = G.number_of_nodes()
        
        # 边界情况处理：如果图空了 (虽然 Done 会处理，但 FeatureExtractor 需健壮)
        if num_nodes == 0:
            return self._empty_data()

        # 2. 构建特征矩阵 X [N, F]
        # Feature 1: Recursive Degree (Normalized by initial max)
        degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
        feat_degree = degrees / self.norm_degree
        
        # Feature 2: Cost (Normalized by initial max)
        costs = np.array([node_costs[n] for n in G.nodes()], dtype=np.float32)
        feat_cost = costs / self.norm_cost
        
        # Stack features: Shape [N, 2]
        x = np.stack([feat_degree, feat_cost], axis=1)
        x_tensor = torch.from_numpy(x).float()
        
        # 3. 构建邻接关系 Edge Index
        # NetworkX edges -> PyG edge_index
        edges = []
        for u, v in G.edges():
            # 将 NX ID 转为 PyG ID
            if u in nx_id_to_pyg_idx and v in nx_id_to_pyg_idx:
                u_idx, v_idx = nx_id_to_pyg_idx[u], nx_id_to_pyg_idx[v]
                edges.append([u_idx, v_idx])
                edges.append([v_idx, u_idx]) # 无向图
        
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. 构建 Action Mask (核心逻辑)
        # 标记哪些节点是“买得起”的。
        # 注意：G 中只包含未删除的节点，所以只需检查 Cost <= Budget
        mask = costs <= budget
        mask_tensor = torch.from_numpy(mask).bool()
        
        # 5. 构建 Global Context (Budget)
        budget_norm = budget / (episode_initial_budget + 1e-5)
        budget_tensor = torch.tensor([[budget_norm]], dtype=torch.float)
        
        # 6. 独立的 Cost Tensor
        costs_tensor = torch.from_numpy(costs).float().unsqueeze(1) / self.norm_cost

        # 封装为 Data 对象
        data = Data(x=x_tensor, 
                    edge_index=edge_index, 
                    costs=costs_tensor,
                    budget=budget_tensor, 
                    mask=mask_tensor)
        
        # ==========================================
        # [CRITICAL FIX] 显式添加 batch 属性
        # 防止模型 forward 时 data.batch 为 None 导致维度错误
        # ==========================================
        data.batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # 挂载辅助信息
        data.mapping = pyg_idx_to_nx_id
        data.num_nodes = num_nodes
        
        return data.to(self.device)

    def _empty_data(self):
        """返回一个空的 Dummy Data，防止程序崩溃"""
        return Data(x=torch.zeros(1, 2), 
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    costs=torch.zeros(1, 1),
                    budget=torch.zeros(1, 1),
                    mask=torch.tensor([False]),
                    mapping={0: 0},
                    num_nodes=0).to(self.device)