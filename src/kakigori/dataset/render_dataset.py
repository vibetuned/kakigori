# Standard library imports
import logging
import zipfile
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third party imports
import cairosvg
from tqdm import tqdm

try:
    # Third party imports
    import verovio
except ImportError:
    logging.warning("verovio is not installed.")

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def render_mxl_to_svg(mxl_path: Path, output_dir: Path, img_dir: Path) -> bool:
    """Render a MusicXML (.mxl) file to SVG files per page and convert to PNG."""
    try:
        verovio.enableLog(False)
        tk = verovio.toolkit()
        tk.setOptions(
            {
                "svgViewBox": True,
                "pageWidth": 2100,  # Add this to ensure consistent page width
                "xmlIdSeed": 42,  # Add this for stable SVG IDs
            }
        )

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
            out_svg = output_dir / f"{mxl_path.stem}_page{page}.svg"
            out_img = img_dir / f"{mxl_path.stem}_page{page}.png"

            if out_svg.exists() and out_img.exists():
                continue

            # Render SVG
            svg = tk.renderToSVG(page)
            out_svg.write_text(svg, encoding="utf-8")

            # Convert to image safely
            try:
                cairosvg.svg2png(
                    url=str(out_svg), write_to=str(out_img), background_color="white"
                )
            except Exception as cairo_e:
                logger.debug(f"CairoSVG failed on {out_svg.name}: {cairo_e}")
                return False

        return True

    except zipfile.BadZipFile:
        pass  # Silently fail on bad zips to keep console clean
    except Exception as e:
        logger.debug(f"Error processing {mxl_path.name}: {e}")

    return False


def _render_target_func(q, m_path, o_dir, i_dir):
    """
    This is the module-level target function for the isolated OS process.
    Because it is at the top level, multiprocessing can pickle it safely.
    """
    # The actual rendering happens inside this completely isolated sub-process
    success = render_mxl_to_svg(m_path, o_dir, i_dir)
    q.put(success)


def _render_isolated(args):
    """
    Spawns an isolated OS process for a single file.
    Catches C++ segfaults, aborts, and infinite loops perfectly.
    """
    mxl_path, output_dir, img_dir = args

    q = multiprocessing.Queue()
    # Point to the globally defined function instead of a local one
    p = multiprocessing.Process(
        target=_render_target_func, args=(q, mxl_path, output_dir, img_dir)
    )
    p.start()

    # 30-second strict timeout to kill infinite rendering loops
    p.join(timeout=30)

    if p.is_alive():
        # The process hung (infinite loop). Kill it.
        p.terminate()
        p.join()
        return mxl_path, False

    if p.exitcode != 0:
        # The process crashed violently (C++ std::terminate, segfault, etc.)
        return mxl_path, False

    # Process finished cleanly, grab the result from the queue
    if not q.empty():
        return mxl_path, q.get()

    return mxl_path, False


def main():
    parser = argparse.ArgumentParser(
        description="Render MXL files to SVG and Image files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .mxl files")
    parser.add_argument(
        "--svg_dir",
        type=str,
        default="data/output_svgs",
        help="Output directory for SVGs",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="data/output_imgs",
        help="Output directory for Images",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    svg_path = Path(args.svg_dir)
    img_path = Path(args.img_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    svg_path.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_path}")
        return

    tasks = [(f, svg_path, img_path) for f in mxl_files]
    success_count = 0
    error_count = 0

    # We use ThreadPoolExecutor to manage the lightweight threads,
    # while the threads handle the heavy, unstable multiprocessing.
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-2) as executor:
        futures = [executor.submit(_render_isolated, task) for task in tasks]

        with tqdm(total=len(mxl_files), desc="Rendering MXL", unit="file") as pbar:
            for future in as_completed(futures):
                mxl_file, is_success = future.result()

                if is_success:
                    success_count += 1
                else:
                    error_count += 1

                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(f"Finished rendering. Success: {success_count} | Failed: {error_count}")


if __name__ == "__main__":
    main()
