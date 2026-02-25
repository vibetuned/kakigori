import xml.etree.ElementTree as ET
from pathlib import Path
import json

def extract_all_classes(svg_dir: str, output_json: str):
    svg_path = Path(svg_dir)
    if not svg_path.exists():
        print(f"Directory not found: {svg_dir}")
        return

    all_classes = set()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    for svg_file in svg_path.rglob("*.svg"):
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            for g in root.findall('.//svg:g', ns):
                cls_str = g.get('class')
                if cls_str:
                    classes = cls_str.split()
                    for cls in classes:
                        if not any(char.isdigit() for char in cls):
                            all_classes.add(cls)
        except Exception as e:
            pass

    classes_list = sorted(list(all_classes))
    with open(output_json, 'w') as f:
        json.dump(classes_list, f, indent=2)

    print(f"Found {len(classes_list)} unique classes. Saved to {output_json}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg_dir", type=str, default="data/output_svgs")
    parser.add_argument("--out", type=str, default="available_classes.json")
    args = parser.parse_args()

    extract_all_classes(args.svg_dir, args.out)

if __name__ == "__main__":
    main()
