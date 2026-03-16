# Standard library imports
import logging
import zipfile
import argparse
import multiprocessing
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def inject_mxl_file(mxl_path: Path, output_dir: Path) -> bool:
    """Injects unique IDs into every XML element of a MusicXML file."""
    try:
        out_path = output_dir / mxl_path.name

        # Don't overwrite if it already exists to allow safe resuming
        if out_path.exists():
            return True

        with zipfile.ZipFile(mxl_path, "r") as zf:
            # Find the main musicxml file
            xml_names = [
                n
                for n in zf.namelist()
                if (n.endswith(".xml") or n.endswith(".musicxml")) and "META" not in n
            ]
            if not xml_names:
                logger.debug(f"No valid XML found inside {mxl_path.name}")
                return False

            xml_name = xml_names[0]
            xml_data = zf.read(xml_name)

            # Preserve META-INF if it exists
            meta_data = (
                zf.read("META-INF/container.xml")
                if "META-INF/container.xml" in zf.namelist()
                else None
            )

        # Parse XML and add IDs to EVERYTHING that doesn't already have one
        try:
            tree = ET.fromstring(xml_data)
        except Exception as e:
            logger.debug(f"XML parsing failed for {mxl_path.name}: {e}")
            return False

        # We use a sequential ID format prefixing with the stem to ensure global uniqueness across the dataset
        # Verovio requires XML valid IDs (must start with letter)
        stem_prefix = "".join([c if c.isalnum() else "" for c in mxl_path.stem[-3:]])
        counter = 1

        for el in tree.iter():
            if "id" not in el.attrib:
                el.attrib["id"] = f"{stem_prefix}-{counter}"
                counter += 1

        # Write back to new zip archive
        new_xml_data = ET.tostring(tree, encoding="unicode")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            zout.writestr(xml_name, new_xml_data)
            if meta_data:
                zout.writestr("META-INF/container.xml", meta_data)

        return True

    except Exception as e:
        logger.debug(f"Failed processing {mxl_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Inject unique IDs into MXL elements to ensure deterministic tracking."
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing original .mxl files"
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        required=True,
        help="Output directory for processed .mxl files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    mxl_files = list(input_dir.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Adding sequential IDs to {len(mxl_files)} MXL files...")

    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {
            executor.submit(inject_mxl_file, path, out_dir): path for path in mxl_files
        }

        with tqdm(total=len(futures), desc="Injecting IDs", unit="file") as pbar:
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
        f"Finished processing. Success: {success_count} | Failed: {error_count}"
    )


if __name__ == "__main__":
    main()
