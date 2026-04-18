# E(3) Equivariant Pipeline pro Structure Based Cofactor Predictor

Tato složka obsahuje paralelní pipeline k trénování a evaluaci využívající **E(3) Steerable GNN** (`model_E3.py`), která je invariantní k rotacím a translacím namísto původního Graph Attention Network (GAT) v `model.py`.

## Co se změnilo
Originální soubory `Binding_site_ex.py` a `binding_site_graph.py` v hlavní složce byly upraveny tak, aby uložily pole souřadnic do atributu `pos` přímo v `torch_geometric.data.Data`. Původní `model.py` tento atribut pouze ignoruje, ale vrstva E(3) ho využívá k výpočtu směrových vektorů a sférických harmonik pro P-L i P-P a L-L interakce.

Tím odpadla nutnost duplikovat extrahovače z PDB souborů a stavitele grafů. Místo toho jsme jen modifikovali hlavní graf abychom zapsali 3D souřadnice. Data processing funguje standardně.

## Odkazy kódů
Tyto skripty místo kompletního přepsání dělají jen "monkey-patching":
- Importují logiku sítě z rodičovské složky.
- Nahrazují třídu `model.GraphClassifier` za Invariantní a Steerable verzi - `model_E3.GraphClassifierE3`.

## Požadavky k tréninku
K běhu originálního E3 Modelu je potřeba instalace repozitáře `Steerable-E3-GNN` dodané z GitHubu (a zejména balíček pro reprezentace grupy SO(3): `e3nn`):

1. Nainstalujte příkazy balíčků v hlavním README zkopírované podsložky (`E3_pipeline/src_e3` atd.) příp. přímo dle repozitáře. 
2. Ověřte, že `import e3nn` nebo `import src_e3` funguje korektně.

## Jak to spustit

Díky těsnému propojení s původními skripty zůstalo původní formátování argumentů stejné (například `manifest_path` jako vstup).

### 1) Sestavení grafů (standardní, ale vytvoří i `pos` parametr)
Probíhá *z kořenového adresáře* klasicky:
```bash
python binding_site_graph.py --pdb-dir PDB ... --output-dir binding_site_graph_batches --split-file train.txt
```

### 2) Trénování E(3) modelu
Z adresáře `E3_pipeline`:
```bash
python train_E3.py --graph-manifest ../binding_site_graph_batches/manifest.json --epochs 5 --hidden-dim 64
```
Všimněte si doporučené defaultní velikosti tenzorové dimenze 64, což chrání proti exponenciálnímu boomu grafových skrytých matic během P-L rotací.

### 3) Evaluace modelu
```bash
python evaluate_E3.py --manifest-path ../binding_site_graph_test/manifest.json --model-path gnn_model_best_e3.pt
```