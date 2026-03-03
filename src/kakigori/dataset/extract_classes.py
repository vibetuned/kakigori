import xml.etree.ElementTree as ET
from pathlib import Path
import json
import re

def extract_all_classes(svg_dir: str, output_json: str):
    svg_path = Path(svg_dir)
    if not svg_path.exists():
        print(f"Directory not found: {svg_dir}")
        return

    all_classes = set()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Load smufl mapping
    smufl_mapping = {}
    try:
        with open('conf/smufl_mapping.json', 'r') as f:
            smufl_mapping = json.load(f)
    except:
        pass

    for svg_file in svg_path.rglob("*.svg"):
        try:
            tree = ET.parse(svg_file)
            root = tree.getroot()
            for g in root.findall('.//svg:g', ns):
                cls_str = g.get('class')
                if cls_str:
                    classes = cls_str.split()
                    
                    if 'barLine' in classes:
                        paths = g.findall('.//svg:path', ns)
                        if len(paths) == 1:
                            if paths[0].get('stroke-dasharray'):
                                classes.append('barlineDashed')
                            else:
                                classes.append('barlineSingle')
                        elif len(paths) >= 2:
                            x_coords = {}
                            for p in paths:
                                d = p.get('d', '')
                                m = re.search(r'M([0-9.]+)', d)
                                if m:
                                    x = round(float(m.group(1)), 1)
                                    if x not in x_coords:
                                        x_coords[x] = []
                                    x_coords[x].append(p)
                            
                            unique_xs = sorted(list(x_coords.keys()))
                            
                            if len(unique_xs) == 1:
                                if x_coords[unique_xs[0]][0].get('stroke-dasharray'):
                                    classes.append('barlineDashed')
                                else:
                                    classes.append('barlineSingle')
                            elif len(unique_xs) == 2:
                                sw1 = float(x_coords[unique_xs[0]][0].get('stroke-width', 1))
                                sw2 = float(x_coords[unique_xs[1]][0].get('stroke-width', 1))
                                if sw1 != sw2:
                                    classes.append('barlineFinal')
                                else:
                                    classes.append('barlineDouble')

                    for cls in classes:
                        if 'timeSig' in cls or not any(char.isdigit() for char in cls):
                            all_classes.add(cls)
            
            parent_map = {c: p for p in root.iter() for c in p}
            for use in root.findall('.//svg:use', ns):
                href = use.get('{http://www.w3.org/1999/xlink}href', '').lstrip('#')
                prefix = href.split('-')[0]
                if prefix in smufl_mapping:
                    specific_label = smufl_mapping[prefix]
                    if specific_label.startswith('accidental'):
                        ancestor = parent_map.get(use)
                        in_keysig = False
                        while ancestor is not None:
                            classes = ancestor.get('class', '').split()
                            if 'keySig' in classes or 'keyAccid' in classes:
                                in_keysig = True
                                break
                            ancestor = parent_map.get(ancestor)
                        if in_keysig:
                            specific_label = specific_label.replace('accidental', 'keyAccid')
                    all_classes.add(specific_label)
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
