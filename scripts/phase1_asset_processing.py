from pathlib import Path
import logging
from datetime import datetime
import argparse # For command-line arguments
import sys # For StreamHandler

from scripts.utils import slide_extractor, image_utils, config

# Get a logger instance for this script
logger = logging.getLogger(__name__)

def setup_logging(day_identifier: str, verbose: bool = False):
    """Sets up logging for this script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_directory = config.LOG_DIR # Use the central log directory

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_name = f"phase1_asset_processing_{day_identifier}_{ts}.log"
    logfile = log_directory / logfile_name

    try:
        log_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # Fallback to console-only logging if directory creation fails
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger.error(f"CRITICAL: Could not create log directory {log_directory}: {e}. Logging to file will fail.")
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)],
    )
    logger.info("Logging initialised for Phase 1 (Asset Processing) -> %s", logfile)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Asset Pre-processing & Extraction for a specific day.")
    parser.add_argument("--day", type=str, required=True, help="Day identifier (e.g., 'day_1', 'day_2')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    day_identifier = args.day

    setup_logging(day_identifier=day_identifier, verbose=args.verbose)

    logger.info(f"=== Phase 1: Asset Pre-processing & Extraction started for {day_identifier} ===")
    if args.verbose:
        logger.debug(f"Verbose logging enabled for Phase 1 script (day: {day_identifier}).")

    # 1. Prepare slide images (crop and adapt to fullscreen)
    logger.info("--- Starting Slide Image Preparation (Cropping & Fullscreen Adaptation) ---")

    # Use day-specific paths from config
    cropped_slides_dir = config.get_cropped_slides_dir(day_identifier)
    raw_slides_dir = config.get_raw_slides_dir(day_identifier)

    if not cropped_slides_dir.exists():
        try:
            cropped_slides_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory for processed slides: %s", cropped_slides_dir)
        except Exception as e:
            logger.error("Failed to create directory %s: %s. Aborting image preparation.", cropped_slides_dir, e)
            # Depending on desired behavior, you might exit or skip this part. For now, just log and continue.
            pass # Allows slide extraction to still run if dir creation fails.

    if cropped_slides_dir.exists(): # Proceed only if directory exists or was created
        if not raw_slides_dir.exists() or not raw_slides_dir.is_dir():
            logger.warning(f"Raw slides directory {raw_slides_dir} does not exist or is not a directory. Skipping image preparation.")
        else:
            for img_path in sorted(raw_slides_dir.iterdir()):
                if (not img_path.is_file()
                        or img_path.suffix.lower() not in config.SUPPORTED_IMAGE_EXTS): # Assuming .jpg etc.
                    continue
                try:
                    # This assumes image_utils.prepare_slide_fullscreen handles saving
                    # and returns the path to the saved processed image or raises an error.
                    processed_image_path = image_utils.prepare_slide_fullscreen(
                        img_path,
                        output_dir=cropped_slides_dir,
                        target_width=config.TARGET_IMAGE_WIDTH,
                        target_height=config.TARGET_IMAGE_HEIGHT,
                        canvas_bg_color=config.CANVAS_BACKGROUND_COLOR
                    )
                    # This block must be inside the try
                    if processed_image_path:
                        logger.info("Prepared (cropped & fullscreened) %s → %s", img_path.name, processed_image_path.relative_to(config.PROJECT_ROOT))
                    else:
                        # This case might not be reached if prepare_slide_fullscreen raises an error on failure,
                        # which is generally a better design.
                        logger.warning("Preparation of %s did not result in an output path or signal success.", img_path.name)
                except Exception as e:
                    logger.error("Failed to prepare slide image %s: %s", img_path.name, e)
    else:
        logger.warning("Skipping slide image preparation as output directory %s could not be created/accessed.", cropped_slides_dir)
    logger.info("--- Slide Image Preparation Finished ---")

    # 2. Extract content via Gemini
    # Note: slide_extractor.process_slides() will need to be aware of the day_identifier
    # if it uses config paths like get_extracted_slide_content_file(day_identifier).
    # This might require passing day_identifier to it, or it reading it from args if run standalone.
    try:
        records = slide_extractor.process_slides(day_identifier) # Assuming process_slides is updated to take day_identifier
        logger.info("Extracted content for %d slides for %s", len(records), day_identifier)
    except Exception as e:
        logger.exception("Slide extraction failed for %s: %s", day_identifier, e)
        sys.exit(1) # Exit with error if slide extraction fails

    logger.info(f"=== Phase 1 for {day_identifier} completed ===")


if __name__ == "__main__":
    # If running standalone, ensure directories are created.
    # The main pipeline (run_pipeline.py) should handle this via config.ensure_directories_exist()
    # before calling phase scripts.
    # For robust standalone execution, this script could also call config.ensure_directories_exist(args.day)
    # after parsing args, but it's omitted here to keep it simpler as it's primarily run by the pipeline.
    main()
