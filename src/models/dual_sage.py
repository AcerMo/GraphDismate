import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class DualStreamSAGE(nn.Module):
    """
    Dual-Stream Architecture for Cost-Sensitive Network Dismantling.
    
    Stream A: Topology Perception (SAGE-Max)
    Stream B: Economic Perception (MLP)
    Fusion  : Gated Attention Mechanism
    """
    def __init__(self, config):
        super(DualStreamSAGE, self).__init__()
        
        # Hyperparameters
        input_dim = 2  # [Norm_Degree, Norm_Cost] (来自 FeatureExtractor)
        hidden_dim = config.get("hidden_dim", 64)
        
        # ================= Stream 1: Topology (GraphSAGE) =================
        # Critical Design: aggr='max'
        # Why? Because dismantling is about finding 'outliers' (hubs), not averages.
        self.sage1 = SAGEConv(input_dim, hidden_dim, aggr='max')
        self.sage2 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.sage3 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        
        # ================= Stream 2: Economic (MLP) =================
        # Input: [Node_Cost, Global_Budget_Ratio]
        # 注意：这里我们提取原始 Cost 特征和 Global Budget 单独处理
        # 这样模型能明确感知"钱"的概念，而不是混在拓扑里
        self.economic_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # ================= Fusion Mechanism (The "Alpha" Gate) =================
        # Input: [Topo_Features, Eco_Features] -> Output: Scalar Alpha (0~1)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.Tanh(), # Tanh often works better for gating intermediate layers
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # ================= Heads =================
        # 1. Actor Head (Policy): Outputs Logits for each node
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Logit
        )
        
        # 2. Critic Head (Value): Outputs State Value V(s)
        # Needs to pool graph-level info
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        """
        Args:
            data (Batch): PyG Batch object.
                - x: [N, 2]
                - edge_index: [2, E]
                - costs: [N, 1] (Normalized cost)
                - budget: [B, 1] (Normalized global budget)
                - batch: [N] (Batch assignments)
        Returns:
            logits: [1, N] (Flattened for distribution sampling)
            value: [B, 1]
            alpha: [N, 1] (For visualization/analysis)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # --- 1. Topology Stream Forward ---
        # Deep SAGE extraction
        h_topo = F.relu(self.sage1(x, edge_index))
        h_topo = F.relu(self.sage2(h_topo, edge_index))
        h_topo = F.relu(self.sage3(h_topo, edge_index)) # [N, hidden]
        
        # --- 2. Economic Stream Forward ---
        # 难点处理：Budget 是 Graph-level 的，Cost 是 Node-level 的
        # 我们需要把 Budget "广播" (Broadcast) 到每个节点上
        
        # data.budget shape: [Batch_Size, 1]
        # data.batch shape: [N]
        # budget_per_node shape: [N, 1]
        budget_per_node = data.budget[batch]
        
        # data.costs shape: [N, 1]
        # Economic Input: [N, 2] -> (Cost, Budget)
        eco_input = torch.cat([data.costs, budget_per_node], dim=1)
        h_eco = self.economic_mlp(eco_input) # [N, hidden]
        
        # --- 3. Gated Fusion ---
        # Concatenate both representations
        combined = torch.cat([h_topo, h_eco], dim=1) # [N, 2*hidden]
        
        # Calculate Alpha
        alpha = self.gate_net(combined) # [N, 1]
        
        # Weighted Sum (Soft Attention)
        # alpha * Topo + (1-alpha) * Eco
        h_final = alpha * h_topo + (1 - alpha) * h_eco # [N, hidden]
        
        # --- 4. Actor Output ---
        node_logits = self.actor_head(h_final) # [N, 1]
        
        # Flatten logits to [1, N] for categorical sampling across the whole batch
        # Note: In PPO, we usually treat the whole batch as one big graph/set of actions
        # But strictly speaking, if batch_size > 1, we need to mask and sample per graph.
        # For simplicity in this implementation, we assume batch_size=1 during inference/rollout
        # Or we reshape later. Here we return [N, 1].
        
        # --- 5. Critic Output ---
        # Global Pooling strategy: Max Pooling over nodes in each graph
        # Why Max? Because the "worst" break in the network (or strongest node) defines state quality.
        # Alternatively: Mean Pooling. Let's use Max for dismantling tasks.
        
        # graph_emb: [Batch_Size, hidden]
        # scatter_max returns (values, indices), we need values [0]
        # If scatter_max is not available (needs torch_scatter), use loop or simple logic
        # Here assuming torch_scatter is installed or using rough approximation:
        
        # Standard PyG global_max_pool
        from torch_geometric.nn import global_max_pool
        graph_emb = global_max_pool(h_final, batch) # [B, hidden]
        
        value = self.critic_mlp(graph_emb) # [B, 1]
        
        return node_logits, value, alpha