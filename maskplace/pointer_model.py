# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# # ---------------- GNN ENCODER ----------------
# class GNNEncoder(nn.Module):
#     def __init__(self, in_dim=4, hidden_dim=128):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self, x, adj):
#         h = F.relu(self.fc1(x))
#         h = torch.matmul(adj, h)   # message passing
#         h = F.relu(self.fc2(h))
#         return h   # [N, hidden_dim]


# # ---------------- POINTER DECODER ----------------
# class PointerDecoder(nn.Module):
#     def __init__(self, hidden_dim=128):
#         super().__init__()

#         self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

#         self.W_q = nn.Linear(hidden_dim, hidden_dim)
#         self.W_k = nn.Linear(hidden_dim, hidden_dim)
#         self.v = nn.Linear(hidden_dim, 1)

#     def forward(self, embeddings):
#         """
#         embeddings: [N, D]
#         """
#         device = embeddings.device
#         N, D = embeddings.shape

#         mask = torch.ones(N, device=device)
#         selected = []
#         log_probs = []

#         # initial hidden state
#         h = torch.zeros(1, D, device=device)
#         c = torch.zeros(1, D, device=device)

#         # initial input = mean embedding
#         inp = embeddings.mean(dim=0).unsqueeze(0)

#         for _ in range(N):

#             h, c = self.lstm(inp, (h, c))

#             # attention scores
#             q = self.W_q(h)                # [1, D]
#             k = self.W_k(embeddings)       # [N, D]

#             scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [N]

#             # apply mask
#             scores = scores + (mask + 1e-8).log()

#             probs = F.softmax(scores, dim=0)

#             dist = torch.distributions.Categorical(probs)
#             idx = dist.sample()

#             log_probs.append(dist.log_prob(idx))
#             selected.append(idx.item())

#             # update mask (NO inplace bug)
#             mask = mask.clone()
#             mask[idx] = 0.0

#             # next input = selected embedding
#             inp = embeddings[idx].unsqueeze(0)

#         return selected, torch.stack(log_probs).sum()


# # ---------------- FULL MODEL ----------------
# class PointerOrderingModel(nn.Module):
#     def __init__(self, in_dim=4, hidden_dim=128):
#         super().__init__()
#         self.encoder = GNNEncoder(in_dim, hidden_dim)
#         self.decoder = PointerDecoder(hidden_dim)

#     def forward(self, x, adj):
#         embeddings = self.encoder(x, adj)
#         return self.decoder(embeddings)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- GAT ENCODER ----------------
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim // heads
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(1, heads, self.out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N]
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.out_dim) # [N, heads, d_out]

        # a_l and a_r: [N, heads, 1]
        a_l = (h * self.attn_l).sum(dim=-1, keepdim=True) 
        a_r = (h * self.attn_r).sum(dim=-1, keepdim=True) 
        
        # FIX: Align dimensions for broadcasting to [N, N, heads]
        # a_l: [N, 1, heads]
        # a_r_t: [1, N, heads]
        a_l = a_l.permute(0, 2, 1) 
        a_r_t = a_r.permute(2, 0, 1) 
        
        # This will broadcast to [N, N, heads]
        score = a_l + a_r_t 
        score = self.leaky_relu(score)

        # Masking (ensure mask is [N, N, 1] or [N, N, heads])
        mask = (adj == 0).unsqueeze(-1)
        score = score.masked_fill(mask, -1e9)
        
        attn = F.softmax(score, dim=1) 
        
        # Aggregate
        # attn: [N, N, heads], h: [N, heads, d_out]
        h_prime = torch.einsum('njh,jhd->nhd', attn, h)
        return h_prime.reshape(N, -1)

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        # 3 Layers of GAT
        self.layer1 = GATLayer(in_dim, hidden_dim)
        self.layer2 = GATLayer(hidden_dim, hidden_dim)
        self.layer3 = GATLayer(hidden_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj):
        # Add self-loops to adj so nodes attend to themselves
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        h = F.elu(self.layer1(x, adj))
        h = F.elu(self.layer2(h, adj)) + h # Residual
        h = self.layer3(h, adj) + h        # Residual
        return self.norm(h)

# ---------------- POINTER DECODER ----------------
class PointerDecoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        
        # Attention components
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        N, D = embeddings.shape
        device = embeddings.device

        mask = torch.zeros(N, device=device) # 0 means available, -inf means masked
        selected_indices = []
        log_probs = []

        # Initialization
        h = torch.zeros(1, D, device=device)
        c = torch.zeros(1, D, device=device)
        inp = embeddings.mean(dim=0, keepdim=True) # Global context start

        for _ in range(N):
            h, c = self.lstm(inp, (h, c))

            # Pointer Attention
            query = self.W_q(h)             # [1, D]
            keys = self.W_k(embeddings)     # [N, D]
            
            # scores: [N]
            scores = self.v(torch.tanh(query + keys)).squeeze(-1)
            
            # Apply mask to prevent re-selecting macros
            masked_scores = scores + mask 

            probs = F.softmax(masked_scores, dim=0)
            
            # Handle numerical stability
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()

            log_probs.append(dist.log_prob(idx))
            selected_indices.append(idx.item())

            # Update mask: set selected to -infinity
            mask[idx] = -1e9
            
            # Feed selected node as next input
            inp = embeddings[idx].unsqueeze(0)

        return selected_indices, torch.stack(log_probs).sum()
    
    def forward_supervised(self, embeddings, target_indices):
        """
        embeddings: [N, D]
        target_indices: [N]  (ground truth ordering indices)

        returns:
            logits: [N, N]
        """
        N, D = embeddings.shape
        device = embeddings.device

        h = torch.zeros(1, D, device=device)
        c = torch.zeros(1, D, device=device)

        inp = embeddings.mean(dim=0, keepdim=True)

        logits = []
        mask = torch.zeros(N, device=device)

        for t in range(N):
            h, c = self.lstm(inp, (h, c))

            query = self.W_q(h)
            keys = self.W_k(embeddings)

            scores = self.v(torch.tanh(query + keys)).squeeze(-1)

            masked_scores = scores + mask
            logits.append(masked_scores)

            # 🔥 teacher forcing
            idx = target_indices[t]
            mask[idx] = -1e9

            inp = embeddings[idx].unsqueeze(0)

        return torch.stack(logits)  # [N, N]

# ---------------- INTEGRATED MODEL ----------------
class PointerOrderingModel(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=128):
        super().__init__()
        self.encoder = GNNEncoder(in_dim, hidden_dim)
        self.decoder = PointerDecoder(hidden_dim)

    def forward(self, x, adj):
        # x: Node features [N, in_dim]
        # adj: Adjacency matrix [N, N]
        embeddings = self.encoder(x, adj)
        return self.decoder(embeddings)
    
    def forward_supervised(self, x, adj, target_indices):
        embeddings = self.encoder(x, adj)
        return self.decoder.forward_supervised(embeddings, target_indices)