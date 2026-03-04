# Standard library imports
import os
import logging
import zipfile
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third party imports
from tqdm import tqdm

try:
    # Third party imports
    import verovio
except ImportError:
    logging.warning("verovio is not installed.")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_mxl(mxl_path: Path, mei_dir: Path, krn_dir: Path) -> tuple[bool, bool]:
    """Convert a MusicXML (.mxl) file to MEI and Humdrum formats."""
    mei_success = False
    krn_success = False
    try:
        verovio.enableLog(verovio.LOG_OFF)
        tk = verovio.toolkit()
        tk.setOptions({"pageWidth": 2100, "xmlIdSeed": 42})

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
                return False, False
            data = zf.read(xml_name).decode("utf-8")

        if not tk.loadData(data):
            return False, False

        out_mei = mei_dir / f"{mxl_path.stem}.mei"
        out_krn = krn_dir / f"{mxl_path.stem}.krn"

        # Render Humdrum
        try:
            if not out_krn.exists():
                # Humdrum export leaks C++ debug logs directly to the OS stdout file descriptor (1).
                # We must redirect fd 1 to /dev/null temporarily.
                fd_null = os.open(os.devnull, os.O_WRONLY)
                old_stdout = os.dup(1)
                os.dup2(fd_null, 1)
                try:
                    tk.getHumdrumFile(str(out_krn))
                finally:
                    os.dup2(old_stdout, 1)
                    os.close(old_stdout)
                    os.close(fd_null)

            krn_success = out_krn.exists()
        except Exception as e:
            logger.debug(f"Humdrum error for {mxl_path.name}: {e}")

        # Render MEI
        try:
            if not out_mei.exists():
                mei_data = tk.getMEI()
                out_mei.write_text(mei_data, encoding="utf-8")
            mei_success = out_mei.exists()
        except Exception as e:
            logger.debug(f"MEI error for {mxl_path.name}: {e}")

        return mei_success, krn_success

    except zipfile.BadZipFile:
        pass  # Silently fail on bad zips to keep console clean
    except Exception as e:
        logger.debug(f"Error processing {mxl_path.name}: {e}")

    return False, False


def _convert_target_func(q, m_path, mei_dir, krn_dir):
    """
    This is the module-level target function for the isolated OS process.
    Because it is at the top level, multiprocessing can pickle it safely.
    """
    mei_success, krn_success = convert_mxl(m_path, mei_dir, krn_dir)
    q.put((mei_success, krn_success))


def _convert_isolated(args):
    """
    Spawns an isolated OS process for a single file.
    Catches C++ segfaults, aborts, and infinite loops perfectly.
    """
    mxl_path, mei_dir, krn_dir = args

    for attempt in range(3):
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(
            target=_convert_target_func, args=(q, mxl_path, mei_dir, krn_dir)
        )
        p.start()

        p.join(timeout=30)

        if p.is_alive():
            p.terminate()
            p.join()
            # print(f"TIMEOUT attempt {attempt}: {mxl_path.name}")
            continue

        if p.exitcode != 0:
            # print(f"CRASH attempt {attempt} (exitcode {p.exitcode}): {mxl_path.name}")
            continue

        if not q.empty():
            return mxl_path, q.get()

    # print(f"FAILED completely: {mxl_path.name}")
    return mxl_path, (False, False)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MXL files to MEI and Humdrum files."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .mxl files")
    parser.add_argument(
        "--mei-dir",
        "--mei_dir",
        dest="mei_dir",
        type=str,
        default="data/output_mei",
        help="Output directory for MEI files",
    )
    parser.add_argument(
        "--krn-dir",
        "--krn_dir",
        dest="krn_dir",
        type=str,
        default="data/output_krn",
        help="Output directory for Humdrum files",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    mei_path = Path(args.mei_dir)
    krn_path = Path(args.krn_dir)

    if not input_path.exists() or not input_path.is_dir():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    mei_path.mkdir(parents=True, exist_ok=True)
    krn_path.mkdir(parents=True, exist_ok=True)

    mxl_files = list(input_path.rglob("*.mxl"))
    if not mxl_files:
        logger.info(f"No .mxl files found in {input_path}")
        return

    tasks = [(f, mei_path, krn_path) for f in mxl_files]
    mei_success_count = 0
    mei_error_count = 0
    krn_success_count = 0
    krn_error_count = 0

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(_convert_isolated, task) for task in tasks]

        with tqdm(total=len(mxl_files), desc="Converting MXL", unit="file") as pbar:
            for future in as_completed(futures):
                mxl_file, (mei_success, krn_success) = future.result()

                if mei_success:
                    mei_success_count += 1
                else:
                    mei_error_count += 1

                if krn_success:
                    krn_success_count += 1
                else:
                    krn_error_count += 1

                pbar.set_postfix(
                    mei_ok=mei_success_count,
                    mei_err=mei_error_count,
                    krn_ok=krn_success_count,
                    krn_err=krn_error_count,
                )
                pbar.update(1)

    logger.info(
        f"Finished converting. MEI (Success: {mei_success_count}, Failed: {mei_error_count}) | Humdrum (Success: {krn_success_count}, Failed: {krn_error_count})"
    )


if __name__ == "__main__":
    main()
