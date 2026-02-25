import argparse
import logging
from pathlib import Path
import json
import re
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

try:
    from svgpathtools import parse_path
except ImportError:
    parse_path = None

import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_target_classes(config_path="gelato_config.json"):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return set(config.get("target_classes", []))
    except Exception as e:
        logger.warning(f"Could not load {config_path}: {e}. Falling back to default classes.")
        return {
            "note", "rest", "clef", "keysig", "metersig", "accid", 
            "artic", "dir", "dynam", "slur", "tie", "hairpin",
            "barline", "beam", "tuplet"
        }

TARGET_CLASSES = load_target_classes()

def get_defs_bboxes(root, ns):
    """Parses SVG <defs> and returns a dictionary mapping glyph IDs to their geometric bounding boxes."""
    defs_bboxes = {}
    if parse_path is None:
        return defs_bboxes

    for g in root.findall('.//svg:defs/svg:g', ns):
        pid = g.get('id')
        path_el = g.find('svg:path', ns)
        if pid and path_el is not None:
            d = path_el.get('d')
            transform = path_el.get('transform', '')
            
            # Verovio often has transform="scale(1,-1)" in defs!
            sy_def = -1 if 'scale(1,-1)' in transform.replace(' ', '') else 1
            
            try:
                path_obj = parse_path(d)
                xmin, xmax, ymin, ymax = path_obj.bbox()
                # Apply the internal def scale
                if sy_def == -1:
                    ymin, ymax = -ymax, -ymin
                defs_bboxes[pid] = (xmin, xmax, ymin, ymax)
            except Exception:
                pass
    return defs_bboxes

def parse_transform_string(transform_str):
    """Extracts translate and scale values from an SVG transform string."""
    tx, ty = 0.0, 0.0
    sx, sy = 1.0, 1.0
    
    if not transform_str:
        return tx, ty, sx, sy

    m_trans = re.search(r'translate\(([^,]+)(?:,\s*([^)]+))?\)', transform_str)
    if m_trans:
        tx = float(m_trans.group(1))
        ty = float(m_trans.group(2)) if m_trans.group(2) else 0.0
        
    m_scale = re.search(r'scale\(([^,]+)(?:,\s*([^)]+))?\)', transform_str)
    if m_scale:
        sx = float(m_scale.group(1))
        sy = float(m_scale.group(2)) if m_scale.group(2) else sx
        
    return tx, ty, sx, sy

def get_absolute_transform(element, parent_map):
    """Calculates the absolute transform of an element by traversing to root."""
    curr = element
    path_nodes = []
    while curr is not None:
        path_nodes.append(curr)
        curr = parent_map.get(curr)
        
    sx, sy, tx, ty = 1.0, 1.0, 0.0, 0.0
    
    # Evaluate from root to element
    for node in reversed(path_nodes):
        transform_str = node.get('transform', '')
        dtx, dty, dsx, dsy = parse_transform_string(transform_str)
        # Apply local translate, then local scale
        tx = sx * dtx + tx
        ty = sy * dty + ty
        sx = sx * dsx
        sy = sy * dsy
        
    return tx, ty, sx, sy

def extract_from_svg(svg_path: Path, img_width: int, img_height: int) -> list:
    """Extracts semantic musical elements and their precise bounding boxes from a Verovio SVG."""
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        logger.debug(f"Failed to parse SVG {svg_path.name}: {e}")
        return []

    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg', 'xlink': 'http://www.w3.org/1999/xlink'}

    # SVG viewBox (Verovio puts the true notation viewBox on an inner SVG, e.g. class="definition-scale")
    inner_svg = root.find('.//svg:svg[@class="definition-scale"]', ns)
    if inner_svg is not None and inner_svg.get('viewBox'):
        viewBox = inner_svg.get('viewBox')
    else:
        viewBox = root.get('viewBox')
        
    if not viewBox:
        return []
    
    try:
        _, _, vw, vh = map(float, viewBox.split())
    except ValueError:
        return []
        
    # Scaling to the PNG image
    scale_x = img_width / vw
    scale_y = img_height / vh

    defs_bboxes = get_defs_bboxes(root, ns)
    parent_map = {c: p for p in root.iter() for c in p}
    annotations = []

    for g in root.findall('.//svg:g', ns):
        cls = g.get('class', '')
        if not cls: continue
        
        # Check if it has a class we care about
        classes = cls.split()
        match = list(TARGET_CLASSES.intersection(classes))
        if not match: continue
        label = match[0]

        # Calculate bounding box from all child <use> and <path> elements
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        # 1. Handle <use> elements (Glyph references)
        for use in g.findall('.//svg:use', ns):
            href = use.get('{http://www.w3.org/1999/xlink}href', '').lstrip('#')
            if href not in defs_bboxes: continue
            
            b_xmin, b_xmax, b_ymin, b_ymax = defs_bboxes[href]
            tx, ty, sx, sy = get_absolute_transform(use, parent_map)
                
            use_xmin = tx + b_xmin * sx
            use_xmax = tx + b_xmax * sx
            use_ymin = ty + b_ymin * sy
            use_ymax = ty + b_ymax * sy
            
            if use_xmin > use_xmax: use_xmin, use_xmax = use_xmax, use_xmin
            if use_ymin > use_ymax: use_ymin, use_ymax = use_ymax, use_ymin

            min_x = min(min_x, use_xmin)
            max_x = max(max_x, use_xmax)
            min_y = min(min_y, use_ymin)
            max_y = max(max_y, use_ymax)

        # 2. Handle native <path> elements (stems, slurs, ledger lines etc.)
        for path in g.findall('.//svg:path', ns):
            d = path.get('d')
            if not d or parse_path is None: continue
            try:
                path_obj = parse_path(d)
                p_xmin, p_xmax, p_ymin, p_ymax = path_obj.bbox()
                
                stroke_width = float(path.get('stroke-width', 0))
                p_xmin -= stroke_width/2
                p_xmax += stroke_width/2
                p_ymin -= stroke_width/2
                p_ymax += stroke_width/2
                
                # Apply absolute transform to native path elements too
                tx, ty, sx, sy = get_absolute_transform(path, parent_map)
                
                # Apply scale then translate
                p_xmin_trans = tx + p_xmin * sx
                p_xmax_trans = tx + p_xmax * sx
                p_ymin_trans = ty + p_ymin * sy
                p_ymax_trans = ty + p_ymax * sy
                
                if p_xmin_trans > p_xmax_trans: p_xmin_trans, p_xmax_trans = p_xmax_trans, p_xmin_trans
                if p_ymin_trans > p_ymax_trans: p_ymin_trans, p_ymax_trans = p_ymax_trans, p_ymin_trans
                
                min_x = min(min_x, p_xmin_trans)
                max_x = max(max_x, p_xmax_trans)
                min_y = min(min_y, p_ymin_trans)
                max_y = max(max_y, p_ymax_trans)
            except Exception:
                pass

        # 3. Handle native <text> elements (tempo markings, lyrics, explicit text)
        for text_el in g.findall('.//svg:text', ns):
            tx, ty, sx, sy = get_absolute_transform(text_el, parent_map)
            
            # Walk every tspan descendant looking for actual text content
            all_elements = [text_el] + list(text_el.iter(f"{{{ns['svg']}}}tspan"))
            
            for el in all_elements:
                if not el.text or not el.text.strip():
                    continue
                
                text_str = el.text.strip()
                    
                # Resolve x/y by walking up ancestors until we find one
                t_x, t_y = 0.0, 0.0
                ancestor = el
                while ancestor is not None:
                    if ancestor.get('x') is not None:
                        t_x = float(ancestor.get('x'))
                        break
                    ancestor = parent_map.get(ancestor)
                ancestor = el
                while ancestor is not None:
                    if ancestor.get('y') is not None:
                        t_y = float(ancestor.get('y'))
                        break
                    ancestor = parent_map.get(ancestor)
                
                # Resolve font-size: walk up ancestors to find the nearest non-zero font-size
                font_size = 100.0  # fallback
                ancestor = el
                while ancestor is not None:
                    fs_str = ancestor.get('font-size')
                    if fs_str:
                        fs_val = float(fs_str.replace('px', '')) if fs_str.endswith('px') else float(fs_str)
                        if fs_val > 0:
                            font_size = fs_val
                            break
                    ancestor = parent_map.get(ancestor)
                
                # Resolve text-anchor from nearest ancestor
                text_anchor = 'start'
                ancestor = el
                while ancestor is not None:
                    ta = ancestor.get('text-anchor')
                    if ta:
                        text_anchor = ta
                        break
                    ancestor = parent_map.get(ancestor)
                
                # Determine font type length coefficient
                is_smufl = any(ord(c) >= 0xE000 for c in text_str)
                char_width = 1.0 if is_smufl else 0.5
                
                width = len(text_str) * font_size * char_width
                height = font_size
                
                # Adjust x based on text-anchor
                if text_anchor == 'middle':
                    min_t_x = t_x - width / 2
                    max_t_x = t_x + width / 2
                elif text_anchor == 'end':
                    min_t_x = t_x - width
                    max_t_x = t_x
                else:  # 'start' or default
                    min_t_x = t_x
                    max_t_x = t_x + width
                    
                min_t_y = t_y - height * 0.8  # Ascender approximation
                max_t_y = t_y + height * 0.2  # Descender approximation
                
                p_xmin_trans = tx + min_t_x * sx
                p_xmax_trans = tx + max_t_x * sx
                p_ymin_trans = ty + min_t_y * sy
                p_ymax_trans = ty + max_t_y * sy
                
                if p_xmin_trans > p_xmax_trans: p_xmin_trans, p_xmax_trans = p_xmax_trans, p_xmin_trans
                if p_ymin_trans > p_ymax_trans: p_ymin_trans, p_ymax_trans = p_ymax_trans, p_ymin_trans
                
                min_x = min(min_x, p_xmin_trans)
                max_x = max(max_x, p_xmax_trans)
                min_y = min(min_y, p_ymin_trans)
                max_y = max(max_y, p_ymax_trans)

        if min_x != float('inf') and min_y != float('inf'):
            # Convert to final image coordinates
            ann = {
                "class": label,
                "bbox": [
                    min_x * scale_x,
                    min_y * scale_y,
                    max_x * scale_x,
                    max_y * scale_y
                ]
            }
            if 'id' in g.attrib:
                ann["id"] = g.attrib['id']
                
            annotations.append(ann)

    return annotations

def process_single_file(svg_path: Path, img_path: Path, out_json: Path) -> bool:
    """Processes a single SVG/PNG pair to extract annotations to JSON."""
    if not svg_path.exists() or not img_path.exists():
        return False
        
    if out_json.exists():
        return True # Skip already processed
        
    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            
        anns = extract_from_svg(svg_path, img_w, img_h)
        if anns:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump({"image": img_path.name, "width": img_w, "height": img_h, "annotations": anns}, f, indent=2)
            return True
            
    except Exception as e:
        logger.debug(f"Failed to process pair {svg_path.name}: {e}")
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Extract bounding box annotations from Verovio SVGs.")
    parser.add_argument("--svg_dir", type=str, default="data/output_svgs", help="Directory containing generated SVGs")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs", help="Directory containing generated Images")
    parser.add_argument("--out_dir", type=str, default="data/annotations", help="Output directory for generated JSON annotations")
    parser.add_argument("--config", type=str, default="gelato_config.json", help="Path to config file for target classes")
    args = parser.parse_args()

    global TARGET_CLASSES
    TARGET_CLASSES = load_target_classes(args.config)

    svg_dir = Path(args.svg_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)

    if not svg_dir.exists() or not img_dir.exists():
        logger.error("SVG or Image directory does not exist.")
        return
        
    if parse_path is None:
        logger.error("svgpathtools is required. Please install it with: uv add svgpathtools")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(img_dir.rglob("*.png"))
    if not img_files:
        logger.info(f"No PNG images found in {img_dir}")
        return

    logger.info(f"Found {len(img_files)} images. Starting annotation extraction...")

    success_count = 0
    error_count = 0
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {}
        for img_file in img_files:
            # Reconstruct the matching SVG path
            svg_file = svg_dir / f"{img_file.stem}.svg"
            out_json = out_dir / f"{img_file.stem}.json"
            futures[executor.submit(process_single_file, svg_file, img_file, out_json)] = img_file
            
        with tqdm(total=len(futures), desc="Extracting Bboxes", unit="file") as pbar:
            for future in as_completed(futures):
                try:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1
                
                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(f"Finished extraction. Successfully processed {success_count}/{len(img_files)} files.")

if __name__ == "__main__":
    main()
