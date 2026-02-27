"""Calculates the median normalized area of every class in the dataset."""

import argparse
import json
import statistics
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-dir", type=str, required=True, help="Path to annotations folder")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    
    # Store a list of all normalized areas for each class
    class_areas = defaultdict(list)

    for json_path in ann_dir.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Get the absolute dimensions of the page
                img_w = data.get("width", 1)
                img_h = data.get("height", 1)
                
                for ann in data.get("annotations", []):
                    cls_name = ann.get("class")
                    bbox = ann.get("bbox")
                    
                    if cls_name and bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        
                        # Calculate normalized width and height (0.0 to 1.0)
                        norm_w = (x2 - x1) / img_w
                        norm_h = (y2 - y1) / img_h
                        
                        # Calculate normalized area
                        area = norm_w * norm_h
                        class_areas[cls_name].append(area)
                        
        except Exception as e:
            print(f"Skipping {json_path.name}: {e}")

    print(f"\n{'CLASS NAME':<25} | {'MEDIAN AREA':<15} | {'RECOMMENDED SCALE (P2, P3, P4)'}")
    print("-" * 75)

    # Calculate median and sort from smallest to largest
    results = []
    for cls_name, areas in class_areas.items():
        median_area = statistics.median(areas)
        results.append((cls_name, median_area))
        
    results.sort(key=lambda x: x[1])

    # Print results and suggest a grid placement
    for cls_name, median_area in results:
        # Dynamic threshold suggestions based on our earlier 0.01 and 0.05 targets
        if median_area < 0.01:
            scale = "P2 (160x160) - Tiny"
        elif median_area < 0.05:
            scale = "P3 (80x80) - Medium"
        else:
            scale = "P4 (40x40) - Large"
            
        print(f"{cls_name:<25} | {median_area:.6f}        | {scale}")

if __name__ == "__main__":
    main()