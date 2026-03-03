# Standard library imports
import logging
import argparse
import multiprocessing
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third party imports
import torch
from tqdm import tqdm

# Local folder imports
from kakigori.graph.parsers import GroundTruthGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_graph(mei_path: Path, json_paths: list, output_dir: Path, node_roles: dict) -> bool:
    """Generate a ground truth graph from MEI and JSON annotations."""
    try:
        if not mei_path.exists() or not json_paths:
            return False
            
        out_graph = output_dir / f"{mei_path.stem}.pt"
        if out_graph.exists():
            return True
            
        # Parse the MEI XML and JSON annotations
        builder = GroundTruthGraphBuilder(str(mei_path), [str(j) for j in json_paths], node_roles)
        gt_edges = builder.build_edges()
        
        # Determine unique nodes present in edges to form the graph
        unique_nodes = list(set([u for u, v, _ in gt_edges] + [v for u, v, _ in gt_edges]))
        
        # In a real model, nodes would have features. For GT generation, we just need the structure.
        # But we must create the edge_index based on indices (0 to N-1) instead of string IDs.
        id_to_idx = {node_id: idx for idx, node_id in enumerate(unique_nodes)}
        
        if not gt_edges:
            # Empty graph? Just save an empty structure.
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_labels = torch.empty(0, dtype=torch.long)
        else:
            # Convert edge pairs from String IDs to PyG tensor index format
            edge_index_list = [[id_to_idx[u], id_to_idx[v]] for u, v, _ in gt_edges]
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            
            # Map labels using parser tool
            edge_labels = builder.get_pyg_labels(edge_index, unique_nodes)

        # Assemble basic PyTorch Geometric Data object dict
        graph_data = {
            "edge_index": edge_index,
            "y": edge_labels,
            "node_ids": unique_nodes,  # Save mapping for later recovery
            "num_nodes": len(unique_nodes),
            "mei_file": mei_path.name,
            "json_files": [j.name for j in json_paths]
        }
        
        # Save output
        torch.save(graph_data, out_graph)
        return True

    except Exception as e:
        logger.debug(f"Error processing {mei_path.name}: {e}")
        
    return False


def _generate_target_func(q, m_path, j_paths, o_dir, node_roles):
    """Module-level target function for isolated process."""
    success = generate_graph(m_path, j_paths, o_dir, node_roles)
    q.put(success)


def _generate_isolated(args):
    """Spawns an isolated OS process with auto-retry."""
    mei_path, json_paths, output_dir, node_roles = args
    
    for attempt in range(3):
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_generate_target_func, args=(q, mei_path, json_paths, output_dir, node_roles))
        p.start()
        
        p.join(timeout=30) 
        
        if p.is_alive():
            p.terminate()
            p.join()
            continue
            
        if p.exitcode != 0:
            continue
            
        if not q.empty():
            return mei_path, q.get()
            
    return mei_path, False


def main():
    parser = argparse.ArgumentParser(description="Generate PyG graphs from MEI and JSON annotations.")
    parser.add_argument("mei_dir", type=str, help="Directory containing .mei files")
    parser.add_argument("json_dir", type=str, help="Directory containing annotation .json files")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", type=str, default="data/output_graphs", help="Output directory for Graph .pt files")
    parser.add_argument("--roles-file", "--roles_file", dest="roles_file", type=str, default="conf/structure.json", help="Path to JSON file containing node roles")
    args = parser.parse_args()

    mei_path = Path(args.mei_dir)
    json_path = Path(args.json_dir)
    out_path = Path(args.out_dir)
    roles_file = Path(args.roles_file)

    if not mei_path.exists() or not mei_path.is_dir():
        logger.error(f"MEI directory does not exist: {mei_path}")
        return
        
    if not json_path.exists() or not json_path.is_dir():
        logger.error(f"JSON annotations directory does not exist: {json_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    mei_files = list(mei_path.rglob("*.mei"))
    if not mei_files:
        logger.info(f"No .mei files found in {mei_path}")
        return

    # Load node roles from JSON
    with open(roles_file, 'r') as f:
        node_roles = json.load(f)["node_roles"]

    # Match MEI files to ALL corresponding annotation JSON files (e.g., _page1, _page2)
    tasks = []
    for m_file in mei_files:
        j_files = list(json_path.glob(f"{m_file.stem}_page*.json"))
        
        if not j_files:
            j_file = json_path / f"{m_file.stem}.json"
            if j_file.exists():
                j_files = [j_file]
                
        if j_files:
            # Sort list so pages are processed iteratively safely
            j_files.sort()
            tasks.append((m_file, j_files, out_path, node_roles))
        else:
            logger.debug(f"Missing json annotation for MEI file: {m_file.name}")

    if not tasks:
        logger.info("Could not pair any MEI files with JSON annotations.")
        return

    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(_generate_isolated, task) for task in tasks]
        
        with tqdm(total=len(tasks), desc="Generating Graphs", unit="graph") as pbar:
            for future in as_completed(futures):
                mei_file, is_success = future.result()
                
                if is_success:
                    success_count += 1
                else:
                    error_count += 1
                
                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(f"Finished generating graphs. Success: {success_count} | Failed: {error_count}")


if __name__ == "__main__":
    main()
