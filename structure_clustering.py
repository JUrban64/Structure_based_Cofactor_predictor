from datasail.sail import datasail
import os
import glob
import shutil
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))


def create_alias_pdb(src_pdb, dst_pdb):
    """Vytvoří symlink, nebo při selhání fyzickou kopii."""
    try:
        os.symlink(src_pdb, dst_pdb)
    except OSError:
        shutil.copy2(src_pdb, dst_pdb)


def cluster_structures():
    pdb_root = os.path.join(script_dir, 'PDB' or 'data_sample')
    pdb_files = glob.glob(os.path.join(pdb_root, '**', '*.pdb'), recursive=True)

    with tempfile.TemporaryDirectory(prefix="datasail_pdb_") as tmp_dir:
        pdb_data = {}

        for pdb_file in pdb_files:
            # Ponecháme pouze base_id bez přípony.
            base_id = os.path.basename(pdb_file).replace(".pdb", "")
            
            if base_id in pdb_data:
                print(f"Warning: duplicate ID '{base_id}', skipping {pdb_file}")
                continue
                
            alias_pdb = os.path.join(tmp_dir, f"{base_id}.pdb")
            create_alias_pdb(pdb_file, alias_pdb)
            pdb_data[base_id] = alias_pdb

        print(f"Loaded {len(pdb_data)} unique PDB structures")
        
        try:
            import json
            with open(os.path.join(script_dir, 'binding_sites_by_protein.json'), 'r') as f:
                bs_data = json.load(f)
            labels_by_pid = {}
            for k, v in bs_data.items():
                labels_by_pid[k] = str(v.get('label', '-1'))
        except Exception:
            labels_by_pid = {}
            
        inter = []
        for base_id in pdb_data.keys():
            inter.append((base_id, labels_by_pid.get(base_id, '-1')))

        try:
            e_splits, f_splits, inter_splits = datasail(
                techniques=["C1e"],
                splits=[8, 1, 1],
                names=["train", "validation", "test"],
                epsilon=0.1,
                e_type="P",
                e_data=pdb_data,
                f_type="O",
                f_data=None,
                inter=inter,
                e_sim="foldseek",
                e_args=" --tmscore-threshold 0.5 -c 0.8",
                solver="SCIP",
            )
            
            # Pojistka pro případ, že DataSAIL vrátí dictionary zabalený v Listu (kvůli parametru 'runs')
            structure_splits = e_splits.get("C1e", {})
            if isinstance(structure_splits, list):
                structure_splits = structure_splits[0]
                
        except Exception as e:
            print(f"Error occurred while running DataSAIL: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

        train, validation, test = [], [], []
        for protein_id, split in structure_splits.items():
            if split == "train":
                train.append(protein_id)
            elif split == "validation":
                validation.append(protein_id)
            elif split == "test":
                test.append(protein_id)

        print(f"Train:      {len(train)} proteins")
        print(f"Validation: {len(validation)} proteins")
        print(f"Test:       {len(test)} proteins")
        
        return train, validation, test


if __name__ == "__main__":
    train, validation, test = cluster_structures()


    with open(os.path.join(script_dir, 'train.txt'), 'w') as f:
        for item in train:
            f.write(f"{item}\n")    

    with open(os.path.join(script_dir, 'validation.txt'), 'w') as f:
        for item in validation:
            f.write(f"{item}\n")
    
    with open(os.path.join(script_dir, 'test.txt'), 'w') as f:
        for item in test:
            f.write(f"{item}\n")