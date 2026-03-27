import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax


class GNNBranch(nn.Module):
    """
    Zpracovává protein-ligand interakční graf.
    
    Varianta C: Kompresovaný ESM v grafu.
      - ESM embeddingy (1280D)

    
    Další vlastnosti:
      - Node type embedding (protein=0, ligand=1)
      - Edge type embedding (P-P=0, P-L=1, L-L=2) 
      - Attention pooling POUZE přes protein uzly
    
    Vstup: PyG Batch s atributy:
        x, edge_index, edge_attr, batch,
        node_type [N], edge_type [E]
    
    Výstup: graph embedding [batch_size, 2*hidden_dim]
    """
    
    # Konstanty
    NUM_NODE_TYPES = 2   # 0=protein, 1=ligand
    NUM_EDGE_TYPES = 3   # 0=PP, 1=PL, 2=LL
    
    def __init__(self, node_dim=1280,
                 hidden_dim=256,
                 num_gnn_layers=3, num_attention_heads=4,
                 dropout=0.5, use_gat=True,
                 ligand_dim=None,
                 esm_dim=1280, esm_compress_dim=64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        self.node_dim = node_dim  # protein feature dim (1280 = ESM + 30) or ligand feature dim (36) - záleží na typu uzlu
        self.esm_dim = esm_dim
        
        if ligand_dim is None:
            # Default ligand feature size from one-hot + chemistry features.
            ligand_dim = 36
        self.ligand_dim = ligand_dim
        

    
        # ---- Protein projection: 1280D → hidden_dim ----
        protein_input_dim = self.esm_dim  # 1280
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- Ligand: 36D → hidden_dim ----
        self.ligand_projection = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- Node type embedding ----
        self.node_type_embedding = nn.Embedding(self.NUM_NODE_TYPES, hidden_dim)
        
        # ---- Edge type embedding ----
        self.edge_type_embedding = nn.Embedding(self.NUM_EDGE_TYPES, hidden_dim)
        
        # ---- GNN layers ----
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_gnn_layers):
            if use_gat:
                self.gnn_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                )
            else:
                self.gnn_layers.append(
                    GCNConv(hidden_dim, hidden_dim)
                )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # ---- Multi-head attention pooling (přes protein nodes) ----
        self.attention_dim = 128
        self.num_heads = num_attention_heads
        self.W1 = nn.Linear(hidden_dim, self.attention_dim)
        self.W2 = nn.Linear(self.attention_dim, num_attention_heads)
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, edge_index, edge_attr, batch, node_type, edge_type):
        # Rozděl protein a ligand uzly
        protein_mask = (node_type == 0)
        ligand_mask = (node_type == 1)
        
        # Projekce proteinů a ligandů
        x_proj = torch.zeros(
            (x.size(0), self.hidden_dim),
            dtype=x.dtype,
            device=x.device,
        )
        # V grafech jsou uzly paddingnute na stejnou dimenzi (typicky 1280),
        # proto bereme jen relevantni cast feature vektoru pro dany typ uzlu.
        x_proj[protein_mask] = self.protein_projection(
            x[protein_mask][:, :self.esm_dim]
        )
        x_proj[ligand_mask] = self.ligand_projection(
            x[ligand_mask][:, :self.ligand_dim]
        )
        
        # Přidej node type embedding
        x_proj += self.node_type_embedding(node_type)
        
        # Přidej edge type embedding (při message passing)
        edge_attr_emb = self.edge_type_embedding(edge_type)
        
        # GNN vrstvy (Pre-Norm & Residual)
        for i, gnn in enumerate(self.gnn_layers):
            h = self.norms[i](x_proj)
            h = gnn(h, edge_index, edge_attr=edge_attr_emb)
            h = torch.relu(h)
            x_proj = h + x_proj if h.shape == x_proj.shape else h
            x_proj = self.dropout(x_proj)
        
        # Attention pooling přes protein nodes
        protein_x = x_proj[protein_mask]
        protein_batch = batch[protein_mask]
        
        # Compute attention scores — normalizace per-graph přes pyg_softmax
        attn_logits = self.W2(torch.tanh(self.W1(protein_x)))  # [N_prot, num_heads]
        # pyg_softmax normalizuje každý head zvlášť v rámci grafu daného uzlu,
        # takže skóre jsou nezávislá na počtu grafů v batchi.
        attn_scores = pyg_softmax(attn_logits, protein_batch, dim=0)  # [N_prot, num_heads]
        
        # Weighted sum pro každý head
        graph_embs = []
        for h in range(self.num_heads):
            weighted_sum = global_add_pool(attn_scores[:, h:h+1] * protein_x, protein_batch)  # [batch_size, hidden_dim]
            graph_embs.append(weighted_sum)
        
        graph_embs = torch.cat(graph_embs, dim=1)  # [batch_size, num_heads*hidden_dim]
        
        return graph_embs


class GraphClassifier(nn.Module):
    """Simple classifier head on top of GNNBranch."""

    def __init__(self, node_dim=1280, hidden_dim=256, num_attention_heads=4,
                 dropout=0.5, num_classes=2, use_gat=True):
        super().__init__()
        self.encoder = GNNBranch(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            use_gat=use_gat,
        )
        self.classifier = nn.Linear(hidden_dim * num_attention_heads, num_classes)

    def forward(self, batch):
        z = self.encoder(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.node_type,
            batch.edge_type,
        )
        return self.classifier(z)
