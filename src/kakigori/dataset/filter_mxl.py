# Standard library imports
import json
import shutil
import logging
import zipfile
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

try:
    # Third party imports
    import verovio
except ImportError:
    logging.warning("verovio is not installed. Filtering will not work.")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_target_classes(config_path: Path) -> set:
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return set(config.get("target_classes", []))
    except Exception as e:
        logger.error(f"Could not load config file {config_path}: {e}")
        return set()


def get_classes_from_svg(svg_content: str) -> set:
    """Parses SVG string and extracts all unique classes."""
    classes = set()
    try:
        root = ET.fromstring(svg_content)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        for g in root.findall(".//svg:g", ns):
            cls_str = g.get("class")
            if cls_str:
                for cls in cls_str.split():
                    # Same logic as extract_classes.py
                    if not any(char.isdigit() for char in cls):
                        classes.add(cls)
    except Exception as e:
        logger.debug(f"Failed to parse SVG content: {e}")
    return classes


def check_mxl_for_classes(mxl_path: Path, target_classes: set) -> bool:
    """Renders MXL to SVG in memory and checks if any target class is present."""
    if not target_classes:
        return False

    try:
        verovio.enableLog(False)
        tk = verovio.toolkit()

        with zipfile.ZipFile(mxl_path, "r") as zf:
            xml_name = next(
                (
                    name
                    for name in zf.namelist()
                    if (name.endswith(".xml") or name.endswith(".musicxml"))
                    and name != "META-INF/container.xml"
                ),
                None,
            )
            if not xml_name:
                return False
            data = zf.read(xml_name).decode("utf-8")

        if not tk.loadData(data):
            return False

        page_count = tk.getPageCount()
        for page in range(1, page_count + 1):
            svg = tk.renderToSVG(page)
            found_classes = get_classes_from_svg(svg)

            if target_classes.intersection(found_classes):
                return True

        return False

    except zipfile.BadZipFile:
        pass
    except Exception as e:
        logger.debug(f"Error checking {mxl_path.name}: {e}")

    return False


def _check_isolated_func(q, m_path, target_classes):
    """Isolated function to prevent verovio crashes from taking down main process."""
    result = check_mxl_for_classes(m_path, target_classes)
    q.put(result)


def _check_isolated(args):
    """Spawns process for single file check."""
    mxl_path, target_classes = args
    q = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_check_isolated_func, args=(q, mxl_path, target_classes)
    )
    p.start()
    p.join(timeout=15)  # 15s timeout

    if p.is_alive():
        p.terminate()
        p.join()
        return mxl_path, False

    if p.exitcode != 0:
        return mxl_path, False

    if not q.empty():
        return mxl_path, q.get()

    return mxl_path, False


def main():
    parser = argparse.ArgumentParser(
        description="Filter MXL files based on required classes."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .mxl files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for matched files",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., conf/config.json)",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=100,
        help="Maximum number of files to copy (default: 100)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    config_path = Path(args.config)

    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    target_classes = load_target_classes(config_path)
    if not target_classes:
        logger.error("No target classes loaded from config. Exiting.")
        return

    logger.info(f"Loaded {len(target_classes)} target classes from {config_path.name}")
    output_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_path}")
        return

    logger.info(
        f"Found {len(mxl_files)} MXL files to process. Target: {args.num_files} files."
    )

    copied_count = 0
    checked_count = 0

    # Standard library imports
    from concurrent.futures import ThreadPoolExecutor

    tasks = [(f, target_classes) for f in mxl_files]

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(_check_isolated, task): task[0] for task in tasks}

        with tqdm(total=args.num_files, desc="Filtering Files", unit="file") as pbar:
            for future in as_completed(futures):
                checked_count += 1
                mxl_path = futures[future]

                try:
                    _, is_match = future.result()

                    if is_match:
                        dest_path = output_path / mxl_path.name
                        if dest_path.exists():
                            dest_path = (
                                output_path
                                / f"{mxl_path.stem}_{mxl_path.parent.name}.mxl"
                            )

                        shutil.copy2(mxl_path, dest_path)
                        copied_count += 1
                        pbar.update(1)
                        logger.debug(f"Match found, copied: {mxl_path.name}")

                        if copied_count >= args.num_files:
                            logger.info(
                                f"\nReached target of {args.num_files} copied files. Stopping early."
                            )
                            # Attempt to cancel remaining pending tasks
                            for f in futures.keys():
                                f.cancel()
                            break

                except Exception as e:
                    logger.debug(f"Failed processing future: {e}")

    logger.info(
        f"Finished. Checked {checked_count} files, copied {copied_count} files to {output_path}"
    )


if __name__ == "__main__":
    main()
