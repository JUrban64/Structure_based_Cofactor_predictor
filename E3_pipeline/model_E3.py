import torch
from torch import nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax

import sys
import os

# Automaticky přidat do cesty lokální repozitář Steerable-E3-GNN pokud existuje
_possible_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src_e3'))
if os.path.isdir(_possible_src) and _possible_src not in sys.path:
    sys.path.insert(0, _possible_src)

try:
    from e3nn.o3 import Irreps, spherical_harmonics
    from models.segnn.segnn import SEGNNLayer
    from models.segnn.o3_building_blocks import O3TensorProduct
    from models.balanced_irreps import WeightBalancedIrreps
except ImportError:
    pass # Necháme volnost pro importní cesty, pokud jsou jinde

class GNNBranchE3(nn.Module):
    """
    E(3)-ekvivariantní verze GNNBranch z model.py využívající Steerable-E3-GNN.
    
    Tento model zachovává rotační a translační symetrie pro 3D souřadnice protein-ligand grafu.
    Očekává: PyG Batch (x, pos, edge_index, batch, node_type, edge_type)
    """
    
    NUM_NODE_TYPES = 2   
    NUM_EDGE_TYPES = 3   
    
    def __init__(self, node_dim=1280,
                 hidden_dim=64, # U E3 snížíme dimenze kvůli paměťové náročnosti tenzorových součinů
                 num_gnn_layers=3, num_attention_heads=4,
                 dropout=0.5,
                 ligand_dim=43,
                 esm_dim=1280, lmax=1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.esm_dim = esm_dim
        self.ligand_dim = ligand_dim
        self.lmax = lmax
        self.num_heads = num_attention_heads
        
        # --- 1. Definice Irreps ---
        # Vstupní features jsou výhradně skaláry (l=0)
        self.scalar_irreps = Irreps(f"{hidden_dim}x0e")
        
        # Geometrické atribucity hran a uzlů (sférické harmoniky do lmax)
        self.attr_irreps = Irreps.spherical_harmonics(lmax)
        
        # Dodatečné zprávy pro hrany (typ hrany a vzdálenost)
        self.edge_emb_dim = 16
        # Vzdálenost (1) + typ hrany (edge_emb_dim) = 17 skalárů
        self.additional_message_irreps = Irreps(f"{1 + self.edge_emb_dim}x0e")
        
        # Kompletní skryté stavy SEGNN
        self.hidden_irreps = WeightBalancedIrreps(
            self.scalar_irreps, self.attr_irreps, sh=True, lmax=lmax
        )

        # --- 2. Projekce vstupů (Skaláry) ---
        self.protein_projection = nn.Sequential(
            nn.Linear(self.esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.ligand_projection = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.node_type_embedding = nn.Embedding(self.NUM_NODE_TYPES, hidden_dim)
        self.edge_type_embedding = nn.Embedding(self.NUM_EDGE_TYPES, self.edge_emb_dim)
        
        # --- 3. Steerable-E3 Vrstvy ---
        # Inicializační tensor product, aby vygeneroval chybějící prvky vyšších řádů ze skalárů
        self.embedding_layer = O3TensorProduct(
            self.scalar_irreps, self.hidden_irreps, self.attr_irreps
        )
        
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                SEGNNLayer(
                    input_irreps=self.hidden_irreps,
                    hidden_irreps=self.hidden_irreps,
                    output_irreps=self.hidden_irreps,
                    edge_attr_irreps=self.attr_irreps,
                    node_attr_irreps=self.attr_irreps,
                    norm="instance",
                    additional_message_irreps=self.additional_message_irreps
                )
            )
            
        # --- 4. Attention pooling ---
        # Z e3nn tensoru extrahujeme zpět pouze skaláry (invariantní k rotacím)
        self.attention_dim = 128
        self.W1 = nn.Linear(hidden_dim, self.attention_dim)
        self.W2 = nn.Linear(self.attention_dim, num_attention_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index, batch, node_type, edge_type):
        protein_mask = (node_type == 0)
        ligand_mask = (node_type == 1)
        
        # A) Inicializace skalárních features
        x_proj = torch.zeros((x.size(0), self.hidden_dim), dtype=x.dtype, device=x.device)
        x_proj[protein_mask] = self.protein_projection(x[protein_mask][:, :self.esm_dim])
        x_proj[ligand_mask] = self.ligand_projection(x[ligand_mask][:, :self.ligand_dim])
        x_proj += self.node_type_embedding(node_type)
        
        # B) Geometrické informace
        # Vektory a vzdálenosti hran
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        
        # Sférické harmoniky vektorů (edge attributes)
        edge_attr_sh = spherical_harmonics(
            self.attr_irreps, rel_pos, normalize=True, normalization='integral'
        )
        
        # Protože modely (např. QM9) většinou definují i node_attr_sh, zprůměrujeme příchozí hrany
        from torch_scatter import scatter
        node_attr_sh = scatter(edge_attr_sh, edge_index[1], dim=0, dim_size=x.size(0), reduce="mean")
        
        # C) Invariantní metainformace na hraně (typ hrany + délka hrany)
        edge_type_inv = self.edge_type_embedding(edge_type)
        additional_messages = torch.cat([edge_dist, edge_type_inv], dim=-1)
        
        # D) Vložení do E(3) ekvivariantního message passingu
        x_e3 = self.embedding_layer(x_proj, node_attr_sh)
        
        for gnn in self.gnn_layers:
            x_e3 = gnn(
                x=x_e3,
                edge_index=edge_index,
                edge_attr=edge_attr_sh,
                node_attr=node_attr_sh,
                batch=batch,
                additional_message_features=additional_messages
            )
            
        # E) Extrakce invariantů a pooling (pozornost se smí počítat jen nad skaláry)
        # Z tensoru [N, vybalancovaná_dimenze] vezmeme prvních hidden_dim hodnot (0e)
        x_inv = x_e3[:, :self.hidden_dim]
        x_inv = self.dropout(x_inv)
        
        protein_x = x_inv[protein_mask]
        protein_batch = batch[protein_mask]
        
        attn_logits = self.W2(torch.tanh(self.W1(protein_x)))
        attn_scores = pyg_softmax(attn_logits, protein_batch, dim=0)
        
        graph_embs = []
        for h in range(self.num_heads):
            weighted_sum = global_add_pool(attn_scores[:, h:h+1] * protein_x, protein_batch, size=batch.max().item() + 1)
            graph_embs.append(weighted_sum)
            
        graph_embs = torch.cat(graph_embs, dim=1) 
        
        return graph_embs

class GraphClassifierE3(nn.Module):
    """
    Classifier využívající Invariantní E(3) Encoder.
    Batch musí obsahovat atribut 'pos' s 3D koordidánatami jako float32.
    """
    def __init__(self, node_dim=1280, hidden_dim=64, num_attention_heads=4, dropout=0.5, num_classes=2):
        super().__init__()
        self.encoder = GNNBranchE3(
            node_dim=node_dim,
            hidden_dim=hidden_dim, # Nižší rozměr, E3 reprezentace bobtná do hloubky
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            lmax=1 # Pro složitější tvary jde zvýšit na 2
        )
        self.classifier = nn.Linear(hidden_dim * num_attention_heads, num_classes)

    def forward(self, batch):
        z = self.encoder(
            batch.x,
            batch.pos,
            batch.edge_index,
            batch.batch,
            batch.node_type,
            batch.edge_type
        )
        return self.classifier(z)
