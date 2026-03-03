"""Class Frequency Counter for Kakigori Dataset.

Scans all annotation JSONs and outputs the exact frequency of every class.
Optionally generates a pruned list of classes for your config file.
"""

import argparse
import json
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Count class frequencies in OMR annotations.")
    parser.add_argument("--ann-dir", type=str, required=True, help="Path to the annotations directory.")
    parser.add_argument("--min-count", type=int, default=50, help="Minimum occurrences required to include a class in the pruned config.")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    if not ann_dir.exists():
        print(f"Error: Directory '{ann_dir}' does not exist.")
        return

    class_counts = Counter()
    total_files = 0
    total_objects = 0

    # Iterate through all JSON files in the directory
    for json_path in ann_dir.rglob("*.json"):
        total_files += 1
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for ann in data.get("annotations", []):
                    cls_name = ann.get("class")
                    if cls_name:
                        class_counts[cls_name] += 1
                        total_objects += 1
        except Exception as e:
            print(f"Warning: Could not read {json_path.name} - {e}")

    # Print Global Stats
    print(f"\nScanned {total_files} files containing {total_objects} total annotated objects.\n")
    print("-" * 40)
    print(f"{'CLASS NAME':<25} | {'FREQUENCY'}")
    print("-" * 40)

    # Print every class sorted from highest frequency to lowest
    for cls_name, count in class_counts.most_common():
        print(f"{cls_name:<25} | {count}")

    print("-" * 40)

    # Generate the pruned config list
    kept_classes = [cls_name for cls_name, count in class_counts.items() if count >= args.min_count]
    kept_classes.sort()  # Sort alphabetically for a clean config file

    print(f"\n✅ Found {len(kept_classes)} classes with {args.min_count} or more occurrences.")
    print(f"Copy the list below into your 'conf/config.json' under 'target_classes':\n")
    print(json.dumps(kept_classes, indent=4))

if __name__ == "__main__":
    main()