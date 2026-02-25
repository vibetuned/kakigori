import argparse
import json
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BBoxVisualizer:
    def __init__(self, master, img_dir: str, ann_dir: str, hierarchy_path: str):
        self.master = master
        self.master.title("Gelato Graph - Dataset Annotations Visualizer")
        
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.hierarchy_path = Path(hierarchy_path)
        
        # Discover all image paths
        self.image_paths = sorted(list(self.img_dir.rglob("*.png")) + list(self.img_dir.rglob("*.jpg")))
        self.current_index = 0
        
        # Load configs
        self.target_classes = self.load_target_classes()
        self.hierarchy = self.load_hierarchy()
        self.visibility_vars = {} # class_name -> tk.BooleanVar
        
        self.setup_ui()
        self.bind_keys()
        
        if self.image_paths:
            self.load_image()
        else:
            logger.warning(f"No images found in {self.img_dir}")
            
    def load_target_classes(self) -> set:
        """Loads the valid target classes from gelato_config.json"""
        target = set()
        if Path("gelato_config.json").exists():
            try:
                with open("gelato_config.json", 'r') as f:
                    data = json.load(f)
                    target = set(data.get("target_classes", []))
            except Exception as e:
                logger.error(f"Failed to read gelato_config.json: {e}")
        return target
        
    def load_hierarchy(self) -> dict:
        """Loads hierarchy.json and populates missing target classes into an 'Uncategorized' group."""
        hierarchy = {}
        if self.hierarchy_path.exists():
            try:
                with open(self.hierarchy_path, 'r') as f:
                    hierarchy = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read {self.hierarchy_path}: {e}")
        
        # Calculate which classes configured in gelato_config are missing from the hierarchy view
        assigned_classes = {cls for group in hierarchy.values() for cls in group}
        missing_classes = self.target_classes - assigned_classes
        
        if missing_classes:
            hierarchy["Uncategorized"] = sorted(list(missing_classes))
            
        # Also clean up groups to only include classes we actually target
        clean_hierarchy = {}
        for group, classes in hierarchy.items():
            valid_classes = [c for c in classes if c in self.target_classes or not self.target_classes]
            if valid_classes:
                clean_hierarchy[group] = valid_classes
                
        return clean_hierarchy

    def setup_ui(self):
        # Base container
        self.main_paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self.sidebar_frame = ttk.Frame(self.main_paned, width=300, padding=10)
        self.main_paned.add(self.sidebar_frame, weight=0)
        
        # Main canvas viewer area
        self.canvas_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.canvas_frame, weight=1)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray20", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bottom Navigation
        self.nav_frame = ttk.Frame(self.canvas_frame, padding=5)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_info = ttk.Label(self.nav_frame, text="0 / 0 : No Images", font=("Arial", 10, "bold"))
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(self.nav_frame, text="Next >", command=self.next_image).pack(side=tk.RIGHT)
        ttk.Button(self.nav_frame, text="< Prev", command=self.prev_image).pack(side=tk.RIGHT)
        
        # Sidebar scrollable area
        ttk.Label(self.sidebar_frame, text="Class Visibility Toggle", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
        
        canvas_sidebar = tk.Canvas(self.sidebar_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.sidebar_frame, orient="vertical", command=canvas_sidebar.yview)
        self.scrollable = ttk.Frame(canvas_sidebar)
        
        self.scrollable.bind("<Configure>", lambda e: canvas_sidebar.configure(scrollregion=canvas_sidebar.bbox("all")))
        canvas_sidebar.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas_sidebar.configure(yscrollcommand=scrollbar.set)
        
        canvas_sidebar.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Build hierarchy checkboxes
        for group_name, classes in self.hierarchy.items():
            group_frame = ttk.Frame(self.scrollable)
            group_frame.pack(fill=tk.X, anchor=tk.W, pady=(5, 0))
            
            group_var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(group_frame, text=group_name, variable=group_var,
                                 command=lambda gvar=group_var, clist=classes: self.toggle_group(gvar, clist))
            cb.pack(anchor=tk.W)
            
            items_frame = ttk.Frame(self.scrollable)
            items_frame.pack(fill=tk.X, anchor=tk.W, padx=(20, 0))
            
            for cls in classes:
                var = tk.BooleanVar(value=True)
                self.visibility_vars[cls] = var
                ttk.Checkbutton(items_frame, text=cls, variable=var, command=self.redraw_image).pack(anchor=tk.W)
                
        self.master.bind("<Configure>", self.on_resize)
        
    def toggle_group(self, group_var, classes):
        state = group_var.get()
        for cls in classes:
            if cls in self.visibility_vars:
                self.visibility_vars[cls].set(state)
        self.redraw_image()

    def bind_keys(self):
        self.master.bind("<Right>", lambda e: self.next_image())
        self.master.bind("<Left>", lambda e: self.prev_image())
        
    def next_image(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.load_image()
            
    def prev_image(self):
        if self.image_paths:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.load_image()
            
    def load_image(self):
        if not self.image_paths: return
        img_path = self.image_paths[self.current_index]
        self.lbl_info.config(text=f"Image {self.current_index + 1} / {len(self.image_paths)} : {img_path.name}")
        
        self.original_image = Image.open(img_path).convert("RGB")
        
        ann_path = self.ann_dir / f"{img_path.stem}.json"
        self.annotations = []
        if ann_path.exists():
            try:
                with open(ann_path, 'r') as f:
                    data = json.load(f)
                    self.annotations = data.get("annotations", [])
            except Exception as e:
                logger.error(f"Error reading {ann_path}: {e}")
        
        self.redraw_image()
        
    def redraw_image(self, event=None):
        if not hasattr(self, 'original_image'): return
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 800
            
        ow, oh = self.original_image.size
        # Letterbox scaling
        scale = min(cw / ow, ch / oh)
        new_w, new_h = int(ow * scale), int(oh * scale)
        if new_w <= 0 or new_h <= 0: return

        resized_img = self.original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(resized_img)
        
        visible_classes = {cls for cls, var in self.visibility_vars.items() if var.get()}
        
        # Generate stable mock colors
        colors = ["red", "blue", "green", "darkorange", "purple", "cyan", "magenta", "gold", "teal", "hotpink"]
        def get_color(cls_name):
            return colors[sum(ord(c) for c in cls_name) % len(colors)]
        
        for ann in self.annotations:
            cls = ann.get('class')
            if cls in visible_classes:
                x1, y1, x2, y2 = ann['bbox']
                sx1, sy1 = x1 * scale, y1 * scale
                sx2, sy2 = x2 * scale, y2 * scale
                
                color = get_color(cls)
                draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
                draw.text((sx1, max(0, sy1 - 10)), cls, fill=color)
                
        self.tk_image = ImageTk.PhotoImage(resized_img)
        self.canvas.delete("all")
        
        x_offset = (cw - new_w) // 2
        y_offset = (ch - new_h) // 2
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.tk_image)
        
    def on_resize(self, event):
        if event.widget == self.canvas:
            self.redraw_image()

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset annotations in an interactive GUI.")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs", help="Directory containing images")
    parser.add_argument("--ann_dir", type=str, default="data/annotations", help="Directory containing annotation JSONs")
    parser.add_argument("--hierarchy", type=str, default="data/hierarchy.json", help="Path to hierarchy.json config")
    args = parser.parse_args()
    
    root = tk.Tk()
    root.geometry("1400x900")
    app = BBoxVisualizer(root, args.img_dir, args.ann_dir, args.hierarchy)
    root.mainloop()

if __name__ == "__main__":
    main()
