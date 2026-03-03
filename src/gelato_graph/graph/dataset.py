import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn import knn_graph

class PianoStaffDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of your raw JSON/XML annotation files and MobileNet outputs
        return ['staff_001.json', 'staff_002.json'] 

    @property
    def processed_file_names(self):
        return ['piano_graphs.pt']

    def process(self):
        data_list = []
        
        for raw_file in self.raw_file_names:
            # 1. Load your MobileNetV5 bounding boxes and classes
            boxes, classes = self._load_mobilenet_outputs(raw_file)
            
            # 2. Build the Node Feature tensor (x) and Position tensor (pos)
            x = self._encode_node_features(boxes, classes)
            pos = torch.tensor([[b.cx, b.cy] for b in boxes], dtype=torch.float)
            
            # 3. Generate Candidate Edges (e.g., connect each node to its 15 nearest neighbors)
            edge_index = knn_graph(pos, k=15, loop=False)
            
            # 4. Assign Ground Truth Labels (y) to the candidate edges based on your XML annotations
            y = self._assign_edge_labels(edge_index, raw_file)
            
            # 5. Construct the PyG Data object
            data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
            data_list.append(data)
            
        self.save(data_list, self.processed_paths[0])