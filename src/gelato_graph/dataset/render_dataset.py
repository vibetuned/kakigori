import argparse
import logging
import zipfile
import verovio
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import cairosvg
from pathlib import Path
from tqdm import tqdm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def render_mxl_to_svg(mxl_path: Path, output_dir: Path, img_dir: Path) -> bool:
    """Render a MusicXML (.mxl) file to SVG files per page and convert to PNG.
    Catches corruption and zip errors gracefully to avoid flooding the stdout.
    """
    # Disable verovio console logging to prevent flooding on corrupt files
    verovio.enableLog(False)
    tk = verovio.toolkit()
    
    # Configure verovio options if needed
    # (these default options ensure standard formatting and ids)
    tk.setOptions({
        "svgViewBox": True,
    })
    
    try:
        # Read MXL as zip first to capture zip defects and avoid segmentation
        with zipfile.ZipFile(mxl_path, 'r') as zf:
            # find the actual xml inside the mxl container
            xml_name = next((name for name in zf.namelist() if (name.endswith('.xml') or name.endswith('.musicxml')) and name != 'META-INF/container.xml'), None)
            if not xml_name:
                logger.debug(f"No valid XML found in {mxl_path.name}")
                return False
            data = zf.read(xml_name).decode('utf-8')

        # Load the raw string data into the verovio toolkit instances
        if not tk.loadData(data):
            logger.debug(f"Verovio failed to load data from {mxl_path.name}")
            return False

        page_count = tk.getPageCount()
        for page in range(1, page_count + 1):
            out_svg = output_dir / f"{mxl_path.stem}_page{page}.svg"
            out_img = img_dir / f"{mxl_path.stem}_page{page}.png"
            
            # Skip only if both SVG and Image exist (if the image folder was populated!)
            if out_svg.exists() and out_img.exists():
                logger.debug(f"Skipping existing page {page} for {mxl_path.name}")
                continue

            svg = tk.renderToSVG(page)
            out_svg.write_text(svg, encoding="utf-8")
            
            # Convert to image with white background
            cairosvg.svg2png(url=str(out_svg), write_to=str(out_img), background_color="white")
            
        logger.debug(f"Rendered {page_count} page(s) for {mxl_path.name}")
        return True

    except zipfile.BadZipFile:
        logger.debug(f"Corrupted MXL file (BadZipFile): {mxl_path.name}")
    except Exception as e:
        logger.debug(f"Error processing {mxl_path.name}: {e}")
        
    return False

def main():
    parser = argparse.ArgumentParser(description="Render MXL files to SVG and Image files.")
    parser.add_argument("input_dir", type=str, help="Directory containing .mxl files")
    parser.add_argument("--svg_dir", type=str, default="data/output_svgs", help="Output directory for generated SVGs")
    parser.add_argument("--img_dir", type=str, default="data/output_imgs", help="Output directory for generated Images")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    svg_path = Path(args.svg_dir)
    img_path = Path(args.img_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_path}")
        return

    svg_path.mkdir(parents=True, exist_ok=True)
    img_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_path}")
        return

    logger.info(f"Found {len(mxl_files)} .mxl files. Starting render...")

    success_count = 0
    
    # Use ProcessPoolExecutor to isolate C++ aborts and parallelize
    # max_workers can be tuned based on CPU cores
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Map futures to filenames for error tracking
        futures = {executor.submit(render_mxl_to_svg, mxl_file, svg_path, img_path): mxl_file for mxl_file in mxl_files}
        
        error_count = 0
        with tqdm(total=len(mxl_files), desc="Rendering MXL", unit="file") as pbar:
            for future in as_completed(futures):
                mxl_file = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    # Catching any process crashes (like C++ terminate aborts)
                    logger.debug(f"Worker process failed for {mxl_file.name}: {e}")
                    error_count += 1
                
                pbar.set_postfix(success=success_count, errors=error_count)
                pbar.update(1)

    logger.info(f"Finished rendering. Successfully processed {success_count}/{len(mxl_files)} files.")

if __name__ == "__main__":
    main()
