Temporal Anchors: These are the nodes that will string together left-to-right to form your chronological Humdrum backbone using your Temporal edges (Class 3).

Structural & Modifiers: These nodes will only ever connect to Temporal Anchors via Structural edges (Class 1) or Modifier edges (Class 2). A stem should never connect directly to a clef.

Context Globals: These dictate the state of the staff (like key signatures and dynamics). In your GNN, these might connect to the measure or staff nodes rather than individual notes.

Virtual Nodes: Since a "sink node" isn't a physical object the PANet detects, it is injected virtually into the PyG data object during the graph-building phase to give the GATv2 a definitive end-of-sequence target.