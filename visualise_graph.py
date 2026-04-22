import argparse
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_geometric.utils import to_networkx, to_dense_adj


def _to_int(value, default=0):
    """Convert tensor/scalar-like values to int for safe comparisons."""
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return int(value.detach().cpu().view(-1)[0].item())
        return int(value)
    except Exception:
        return default


def _load_graph_from_path(input_path, index=0):
    """Load one PyG Data graph from a .pt file produced by this project."""
    payload = torch.load(input_path, map_location='cpu', weights_only=False)

    if isinstance(payload, dict) and 'graphs' in payload:
        graphs = payload['graphs']
    elif isinstance(payload, list):
        graphs = payload
    else:
        # Already a single graph-like object
        return payload

    if not graphs:
        raise ValueError(f"Soubor {input_path} neobsahuje žádné grafy.")
    if index < 0 or index >= len(graphs):
        raise IndexError(
            f"Index grafu {index} je mimo rozsah 0..{len(graphs) - 1}"
        )
    return graphs[index]


def visualize_binding_site_graph(data, output_path='binding_site_graph_visualization.png', show=False):
    """
    Vizualizuje PyG Data objekt představující binding site.
    Vykreslí 2D reprezentaci grafu a Adjacency Matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ==========================================
    # 1. VIZUALIZACE GRAFU (TOPOLOGIE)
    # ==========================================
    ax_graph = axes[0]
    
    # Převod PyG grafu do NetworkX
    # node_attrs a edge_attrs zajistí, že se přenesou typy
    G = to_networkx(data, node_attrs=['node_type'], edge_attrs=['edge_type'], to_undirected=True)
    
    # Definice barev pro uzly (0 = Protein, 1 = Ligand)
    node_colors = []
    for _, node_data in G.nodes(data=True):
        ntype = _to_int(node_data.get('node_type', 0), default=0)
        if ntype == 0:
            node_colors.append('skyblue')     # Protein (Modrá)
        else:
            node_colors.append('lightgreen')  # Ligand (Zelená)
            
    # Definice barev pro hrany (0 = P-P, 1 = P-L, 2 = L-L)
    edge_colors = []
    for _, _, edge_data in G.edges(data=True):
        etype = _to_int(edge_data.get('edge_type', 0), default=0)
        if etype == 0:
            edge_colors.append('lightgray')   # P-P
        elif etype == 1:
            edge_colors.append('orange')      # P-L (Interakce)
        else:
            edge_colors.append('darkgreen')   # L-L (Kovalentní)

    # Vypočítáme pozice uzlů pomocí silově orientovaného algoritmu (spring layout)
    # k=0.15 určuje optimální vzdálenost mezi uzly
    pos = nx.spring_layout(G, k=0.15, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors, edgecolors='black', node_size=400)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color=edge_colors, width=2.0)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=9, font_color='black')
    
    # Legenda pro graf
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=12, label='Protein Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=12, label='Ligand Node'),
        Line2D([0], [0], color='lightgray', lw=3, label='P-P Edge (Contact)'),
        Line2D([0], [0], color='orange', lw=3, label='P-L Edge (Interaction)'),
        Line2D([0], [0], color='darkgreen', lw=3, label='L-L Edge (Bond)')
    ]
    ax_graph.legend(handles=legend_elements, loc='upper left')
    ax_graph.set_title("Struktura Binding Site Grafu", fontsize=14, fontweight='bold')
    ax_graph.axis('off')

    # ==========================================
    # 2. VIZUALIZACE ADJACENCY MATRIX
    # ==========================================
    ax_mat = axes[1]
    
    # Převedeme edge_index a edge_type na hustou matici [N, N]
    # Kde není hrana, bude 0. Abychom odlišili typy hran (0, 1, 2), 
    # posuneme hodnoty typů hran o +1 (1=P-P, 2=P-L, 3=L-L). 0 bude "bez hrany".
    edge_type_shifted = data.edge_type + 1
    dense_adj = to_dense_adj(data.edge_index, edge_attr=edge_type_shifted)[0].detach().cpu().numpy()
    
    # Vlastní colormap: 0=Bílá(prázdno), 1=Šedá(P-P), 2=Oranžová(P-L), 3=Zelená(L-L)
    cmap = ListedColormap(['white', 'lightgray', 'orange', 'darkgreen'])
    
    cax = ax_mat.matshow(dense_adj, cmap=cmap, vmin=0, vmax=3)
    
    # Zjištění počtu proteinových uzlů (pro nakreslení dělící čáry)
    num_protein_nodes = (data.node_type == 0).sum().item()
    
    # Nakreslení červených čar, které matici vizuálně rozdělí na 4 bloky
    if num_protein_nodes > 0 and num_protein_nodes < data.num_nodes:
        # Odčítáme 0.5, protože matshow kreslí čtverečky na celočíselné souřadnice se středem
        split_idx = num_protein_nodes - 0.5
        ax_mat.axhline(y=split_idx, color='red', linestyle='--', linewidth=2, label='Rozhraní P / L')
        ax_mat.axvline(x=split_idx, color='red', linestyle='--', linewidth=2)
        ax_mat.legend(loc='upper right')
        
    ax_mat.set_title("Adjacency Matrix (Sousednost uzlů)", fontsize=14, fontweight='bold', pad=20)
    ax_mat.set_xlabel("Cílový uzel (Index)", fontsize=12)
    ax_mat.set_ylabel("Zdrojový uzel (Index)", fontsize=12)
    ax_mat.grid(False)
    
    # Převedeme osy X do spodní části (matshow je dává nahoru)
    ax_mat.xaxis.set_ticks_position('bottom')
    
    # Přidání barevné škály (Colorbar) k matici
    cbar = fig.colorbar(cax, ax=ax_mat, ticks=[0.375, 1.125, 1.875, 2.625], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(['Bez hrany', 'P-P (Kontaktní)', 'P-L (Interakce)', 'L-L (Kovalentní)'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Uloží vizualizaci jako PNG
    if show:
        plt.show()
    plt.close(fig)

# ==========================================
# PŘÍKLAD POUŽITÍ S TVÝM KÓDEM:
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vizualizace binding-site grafu z .pt souboru')
    parser.add_argument('--input', required=True, help='Cesta k .pt souboru (list grafů nebo dict s klíčem graphs)')
    parser.add_argument('--index', type=int, default=0, help='Index grafu v souboru (default: 0)')
    parser.add_argument('--output', default='binding_site_graph_visualization.png', help='Výstupní PNG cesta')
    parser.add_argument('--show', action='store_true', help='Zobrazit graf interaktivně přes matplotlib')
    args = parser.parse_args()

    graph = _load_graph_from_path(args.input, index=args.index)
    visualize_binding_site_graph(graph, output_path=args.output, show=args.show)
    print(f'Vizualizace uložena do: {args.output}')