# Standard library imports
import re
import json
import math
import logging
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from PIL import Image
from tqdm import tqdm

try:
    # Third party imports
    from svgpathtools import parse_path
except ImportError:
    parse_path = None

# Standard library imports
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_target_classes(config_path="conf/config.json"):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return set(config.get("target_classes", []))
    except Exception as e:
        logger.warning(
            f"Could not load {config_path}: {e}. Falling back to default classes."
        )
        return {
            "note",
            "rest",
            "clef",
            "keysig",
            "metersig",
            "accid",
            "artic",
            "dir",
            "dynam",
            "slur",
            "tie",
            "hairpin",
            "barline",
            "beam",
            "tuplet",
        }


def load_smufl_mapping(mapping_path="conf/smufl_mapping.json"):
    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {mapping_path}: {e}")
        return {}


def get_defs_bboxes(root, ns):
    """Parses SVG <defs> and returns a dictionary mapping glyph IDs to their geometric bounding boxes."""
    defs_bboxes = {}
    if parse_path is None:
        return defs_bboxes

    for g in root.findall(".//svg:defs/svg:g", ns):
        pid = g.get("id")
        path_el = g.find("svg:path", ns)
        if pid and path_el is not None:
            d = path_el.get("d")
            transform = path_el.get("transform", "")

            # Verovio often has transform="scale(1,-1)" in defs!
            sy_def = -1 if "scale(1,-1)" in transform.replace(" ", "") else 1

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


def multiply_matrices(m1, m2):
    """Multiplies two SVG transform matrices (a, b, c, d, e, f)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2

    a = a1 * a2 + c1 * b2
    b = b1 * a2 + d1 * b2
    c = a1 * c2 + c1 * d2
    d = b1 * c2 + d1 * d2
    e = a1 * e2 + c1 * f2 + e1
    f = b1 * e2 + d1 * f2 + f1
    return (a, b, c, d, e, f)


def apply_transform_to_bbox(matrix, xmin, xmax, ymin, ymax):
    """Applies an affine transformation matrix to a bounding box."""
    a, b, c, d, e, f = matrix
    corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    xs = [a * x + c * y + e for x, y in corners]
    ys = [b * x + d * y + f for x, y in corners]
    return min(xs), max(xs), min(ys), max(ys)


def parse_transform_string(transform_str):
    """Extracts the transformation matrix from an SVG transform string."""
    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    if not transform_str:
        return matrix

    commands = re.findall(
        r"(translate|scale|rotate|matrix)\s*\(([^)]+)\)", transform_str
    )

    for cmd, args_str in commands:
        args = [float(val) for val in re.split(r"[, ]+", args_str.strip()) if val]
        if not args:
            continue

        if cmd == "translate":
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            cmd_matrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif cmd == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            cmd_matrix = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif cmd == "rotate":
            angle = args[0]
            cx = args[1] if len(args) > 1 else 0.0
            cy = args[2] if len(args) > 2 else 0.0

            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)

            rot_m = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)

            if cx != 0.0 or cy != 0.0:
                t1 = (1.0, 0.0, 0.0, 1.0, cx, cy)
                t2 = (1.0, 0.0, 0.0, 1.0, -cx, -cy)
                cmd_matrix = multiply_matrices(multiply_matrices(t1, rot_m), t2)
            else:
                cmd_matrix = rot_m
        elif cmd == "matrix":
            if len(args) >= 6:
                cmd_matrix = tuple(args[:6])
            else:
                continue
        else:
            continue

        matrix = multiply_matrices(matrix, cmd_matrix)

    return matrix


def get_absolute_transform(element, parent_map):
    """Calculates the absolute transform of an element by traversing to root."""
    curr = element
    path_nodes = []
    while curr is not None:
        path_nodes.append(curr)
        curr = parent_map.get(curr)

    matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    # Evaluate from root to element
    for node in reversed(path_nodes):
        transform_str = node.get("transform", "")
        local_matrix = parse_transform_string(transform_str)
        matrix = multiply_matrices(matrix, local_matrix)

    return matrix


def extract_from_svg(
    svg_path: Path,
    img_width: int,
    img_height: int,
    target_classes: set,
    smufl_mapping: dict,
) -> list:
    """Extracts semantic musical elements and their precise bounding boxes from a Verovio SVG."""
    try:
        tree = ET.parse(svg_path)
    except Exception as e:
        logger.debug(f"Failed to parse SVG {svg_path.name}: {e}")
        return []

    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}

    # SVG viewBox (Verovio puts the true notation viewBox on an inner SVG, e.g. class="definition-scale")
    inner_svg = root.find('.//svg:svg[@class="definition-scale"]', ns)
    if inner_svg is not None and inner_svg.get("viewBox"):
        viewBox = inner_svg.get("viewBox")
    else:
        viewBox = root.get("viewBox")

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

    for g in root.findall(".//svg:g", ns):
        cls = g.get("class", "")
        if not cls:
            continue

        # Check if it has a class we care about
        classes = cls.split()

        # Heuristic for barLine
        if "barLine" in classes:
            paths = g.findall(".//svg:path", ns)
            if len(paths) == 1:
                if paths[0].get("stroke-dasharray"):
                    classes.append("barlineDashed")
                else:
                    classes.append("barlineSingle")
            elif len(paths) >= 2:
                # Group paths by x-coordinate to properly handle multi-staff barlines
                x_coords = {}
                for p in paths:
                    d = p.get("d", "")
                    m = re.search(r"M([0-9.]+)", d)
                    if m:
                        x = round(float(m.group(1)), 1)
                        if x not in x_coords:
                            x_coords[x] = []
                        x_coords[x].append(p)

                unique_xs = sorted(list(x_coords.keys()))

                if len(unique_xs) == 1:
                    # It's actually a single barline spanning multiple staves
                    # Check for dasharray on the first path
                    if x_coords[unique_xs[0]][0].get("stroke-dasharray"):
                        classes.append("barlineDashed")
                    else:
                        classes.append("barlineSingle")
                elif len(unique_xs) == 2:
                    # Double or Final
                    sw1 = float(x_coords[unique_xs[0]][0].get("stroke-width", 1))
                    sw2 = float(x_coords[unique_xs[1]][0].get("stroke-width", 1))
                    if sw1 != sw2:
                        classes.append("barlineFinal")
                    else:
                        classes.append("barlineDouble")

        match = list(target_classes.intersection(classes))
        if not match:
            continue
        label = match[0]

        # Calculate bounding box from all child <use> and <path> elements
        min_x, max_x, min_y, max_y = (
            float("inf"),
            float("-inf"),
            float("inf"),
            float("-inf"),
        )

        # 1. Handle <use> elements (Glyph references)
        for use in g.findall(".//svg:use", ns):
            href = use.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
            if href not in defs_bboxes:
                continue

            b_xmin, b_xmax, b_ymin, b_ymax = defs_bboxes[href]
            matrix = get_absolute_transform(use, parent_map)

            use_xmin, use_xmax, use_ymin, use_ymax = apply_transform_to_bbox(
                matrix, b_xmin, b_xmax, b_ymin, b_ymax
            )

            min_x = min(min_x, use_xmin)
            max_x = max(max_x, use_xmax)
            min_y = min(min_y, use_ymin)
            max_y = max(max_y, use_ymax)

        # 2. Handle native <path> elements (stems, slurs, ledger lines etc.)
        for path in g.findall(".//svg:path", ns):
            d = path.get("d")
            if not d or parse_path is None:
                continue
            try:
                path_obj = parse_path(d)
                p_xmin, p_xmax, p_ymin, p_ymax = path_obj.bbox()

                stroke_width = float(path.get("stroke-width", 0))
                p_xmin -= stroke_width / 2
                p_xmax += stroke_width / 2
                p_ymin -= stroke_width / 2
                p_ymax += stroke_width / 2

                # Apply absolute transform to native path elements too
                matrix = get_absolute_transform(path, parent_map)

                p_xmin_trans, p_xmax_trans, p_ymin_trans, p_ymax_trans = (
                    apply_transform_to_bbox(matrix, p_xmin, p_xmax, p_ymin, p_ymax)
                )

                min_x = min(min_x, p_xmin_trans)
                max_x = max(max_x, p_xmax_trans)
                min_y = min(min_y, p_ymin_trans)
                max_y = max(max_y, p_ymax_trans)
            except Exception:
                pass

        # 3. Handle <rect> elements
        for rect in g.findall(".//svg:rect", ns):
            try:
                x = float(rect.get("x", 0))
                y = float(rect.get("y", 0))
                w = float(rect.get("width", 0))
                h = float(rect.get("height", 0))

                if w > 0 and h > 0:
                    r_xmin, r_xmax = x, x + w
                    r_ymin, r_ymax = y, y + h

                    matrix = get_absolute_transform(rect, parent_map)
                    r_xmin_trans, r_xmax_trans, r_ymin_trans, r_ymax_trans = (
                        apply_transform_to_bbox(matrix, r_xmin, r_xmax, r_ymin, r_ymax)
                    )

                    min_x = min(min_x, r_xmin_trans)
                    max_x = max(max_x, r_xmax_trans)
                    min_y = min(min_y, r_ymin_trans)
                    max_y = max(max_y, r_ymax_trans)
            except Exception:
                pass

        # 4. Handle <line> elements
        for line in g.findall(".//svg:line", ns):
            try:
                x1 = float(line.get("x1", 0))
                y1 = float(line.get("y1", 0))
                x2 = float(line.get("x2", 0))
                y2 = float(line.get("y2", 0))

                stroke_width = float(line.get("stroke-width", 0))
                pad = stroke_width / 2

                l_xmin, l_xmax = min(x1, x2) - pad, max(x1, x2) + pad
                l_ymin, l_ymax = min(y1, y2) - pad, max(y1, y2) + pad

                matrix = get_absolute_transform(line, parent_map)
                l_xmin_trans, l_xmax_trans, l_ymin_trans, l_ymax_trans = (
                    apply_transform_to_bbox(matrix, l_xmin, l_xmax, l_ymin, l_ymax)
                )

                min_x = min(min_x, l_xmin_trans)
                max_x = max(max_x, l_xmax_trans)
                min_y = min(min_y, l_ymin_trans)
                max_y = max(max_y, l_ymax_trans)
            except Exception:
                pass

        # 5. Handle native <text> elements (tempo markings, lyrics, explicit text)
        for text_el in g.findall(".//svg:text", ns):
            matrix = get_absolute_transform(text_el, parent_map)

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
                    if ancestor.get("x") is not None:
                        t_x = float(ancestor.get("x"))
                        break
                    ancestor = parent_map.get(ancestor)
                ancestor = el
                while ancestor is not None:
                    if ancestor.get("y") is not None:
                        t_y = float(ancestor.get("y"))
                        break
                    ancestor = parent_map.get(ancestor)

                # Resolve font-size: walk up ancestors to find the nearest non-zero font-size
                font_size = 100.0  # fallback
                ancestor = el
                while ancestor is not None:
                    fs_str = ancestor.get("font-size")
                    if fs_str:
                        fs_val = (
                            float(fs_str.replace("px", ""))
                            if fs_str.endswith("px")
                            else float(fs_str)
                        )
                        if fs_val > 0:
                            font_size = fs_val
                            break
                    ancestor = parent_map.get(ancestor)

                # Resolve text-anchor from nearest ancestor
                text_anchor = "start"
                ancestor = el
                while ancestor is not None:
                    ta = ancestor.get("text-anchor")
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
                if text_anchor == "middle":
                    min_t_x = t_x - width / 2
                    max_t_x = t_x + width / 2
                elif text_anchor == "end":
                    min_t_x = t_x - width
                    max_t_x = t_x
                else:  # 'start' or default
                    min_t_x = t_x
                    max_t_x = t_x + width

                min_t_y = t_y - height * 0.8  # Ascender approximation
                max_t_y = t_y + height * 0.2  # Descender approximation

                p_xmin_trans, p_xmax_trans, p_ymin_trans, p_ymax_trans = (
                    apply_transform_to_bbox(matrix, min_t_x, max_t_x, min_t_y, max_t_y)
                )

                min_x = min(min_x, p_xmin_trans)
                max_x = max(max_x, p_xmax_trans)
                min_y = min(min_y, p_ymin_trans)
                max_y = max(max_y, p_ymax_trans)

        # 6. Handle <ellipse> elements
        for ellipse in g.findall(".//svg:ellipse", ns):
            try:
                cx = float(ellipse.get("cx", 0))
                cy = float(ellipse.get("cy", 0))
                rx = float(ellipse.get("rx", 0))
                ry = float(ellipse.get("ry", 0))

                if rx > 0 and ry > 0:
                    e_xmin, e_xmax = cx - rx, cx + rx
                    e_ymin, e_ymax = cy - ry, cy + ry

                    matrix = get_absolute_transform(ellipse, parent_map)
                    e_xmin_trans, e_xmax_trans, e_ymin_trans, e_ymax_trans = (
                        apply_transform_to_bbox(matrix, e_xmin, e_xmax, e_ymin, e_ymax)
                    )

                    min_x = min(min_x, e_xmin_trans)
                    max_x = max(max_x, e_xmax_trans)
                    min_y = min(min_y, e_ymin_trans)
                    max_y = max(max_y, e_ymax_trans)
            except Exception:
                pass

        if min_x != float("inf") and min_y != float("inf"):
            # Convert to final image coordinates
            ann = {
                "class": label,
                "bbox": [
                    min_x * scale_x,
                    min_y * scale_y,
                    max_x * scale_x,
                    max_y * scale_y,
                ],
            }
            if "id" in g.attrib:
                ann["id"] = g.attrib["id"]

            annotations.append(ann)

    # 7. Global <use> elements for specific SMuFL classes
    for use in root.findall(".//svg:use", ns):
        href = use.get("{http://www.w3.org/1999/xlink}href", "").lstrip("#")
        prefix = href.split("-")[0]
        specific_label = smufl_mapping.get(prefix)
        if not specific_label:
            continue

        # Contextual override: if the mapping is an accidental but it's inside a keySig/keyAccid group
        if specific_label.startswith("accidental"):
            ancestor = parent_map.get(use)
            in_keysig = False
            while ancestor is not None:
                classes = ancestor.get("class", "").split()
                if "keySig" in classes or "keyAccid" in classes:
                    in_keysig = True
                    break
                ancestor = parent_map.get(ancestor)

            if in_keysig:
                specific_label = specific_label.replace("accidental", "keyAccid")

        if specific_label not in target_classes:
            continue

        if href not in defs_bboxes:
            continue

        b_xmin, b_xmax, b_ymin, b_ymax = defs_bboxes[href]
        matrix = get_absolute_transform(use, parent_map)
        use_xmin, use_xmax, use_ymin, use_ymax = apply_transform_to_bbox(
            matrix, b_xmin, b_xmax, b_ymin, b_ymax
        )

        ann = {
            "class": specific_label,
            "bbox": [
                use_xmin * scale_x,
                use_ymin * scale_y,
                use_xmax * scale_x,
                use_ymax * scale_y,
            ],
        }

        # Determine ID from parent
        ancestor = parent_map.get(use)
        while ancestor is not None:
            if "id" in ancestor.attrib and ancestor.get("class"):
                ann["id"] = ancestor.attrib["id"]
                break
            ancestor = parent_map.get(ancestor)

        annotations.append(ann)

    return annotations


def process_single_file(
    svg_path: Path,
    img_path: Path,
    out_json: Path,
    target_classes: set,
    smufl_mapping: dict,
) -> bool:
    """Processes a single SVG/PNG pair to extract annotations to JSON."""
    if not svg_path.exists() or not img_path.exists():
        return False

    if out_json.exists():
        return True  # Skip already processed

    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        anns = extract_from_svg(svg_path, img_w, img_h, target_classes, smufl_mapping)
        if anns:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image": img_path.name,
                        "width": img_w,
                        "height": img_h,
                        "annotations": anns,
                    },
                    f,
                    indent=2,
                )
            return True

    except Exception as e:
        logger.debug(f"Failed to process pair {svg_path.name}: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract bounding box annotations from Verovio SVGs."
    )
    parser.add_argument(
        "--svg_dir",
        type=str,
        default="data/output_svgs",
        help="Directory containing generated SVGs",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="data/output_imgs",
        help="Directory containing generated Images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/annotations",
        help="Output directory for generated JSON annotations",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.json",
        help="Path to config file for target classes",
    )
    args = parser.parse_args()

    target_classes = load_target_classes(args.config)
    smufl_mapping = load_smufl_mapping()

    svg_dir = Path(args.svg_dir)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)

    if not svg_dir.exists() or not img_dir.exists():
        logger.error("SVG or Image directory does not exist.")
        return

    if parse_path is None:
        logger.error(
            "svgpathtools is required. Please install it with: uv add svgpathtools"
        )
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
            futures[
                executor.submit(
                    process_single_file,
                    svg_file,
                    img_file,
                    out_json,
                    target_classes,
                    smufl_mapping,
                )
            ] = img_file

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

    logger.info(
        f"Finished extraction. Successfully processed {success_count}/{len(img_files)} files."
    )


if __name__ == "__main__":
    main()
