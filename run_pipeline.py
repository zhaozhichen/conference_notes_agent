import subprocess
import sys
import logging
from pathlib import Path
import argparse # For command-line arguments
from datetime import datetime

# This assumes that the 'scripts' directory is in the same directory as 'run_pipeline.py'
# and that the project root is where 'run_pipeline.py' resides.
# If not, config import might need adjustment or PROJECT_ROOT in config.py needs to be robust.
try:
    from scripts.utils import config # To access configured paths if needed directly
except ImportError:
    # Cannot use logger here as it's not configured yet. Print to stderr.
    print("CRITICAL: Failed to import config from scripts.utils. Ensure __init__.py exists and PYTHONPATH is correct if running from outside project root.", file=sys.stderr)
    print("CRITICAL: Attempting to add project root to sys.path for robust module loading.", file=sys.stderr)
    project_root = Path(__file__).resolve().parent
    scripts_path = project_root / "scripts"
    # Add project root and scripts path to sys.path to help find the config module
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(scripts_path) not in sys.path: # Though -m handles this better
         sys.path.insert(0, str(scripts_path))
    try:
        from scripts.utils import config
    except ImportError as e:
        print(f"CRITICAL: Still failed to import config after path adjustment: {e}", file=sys.stderr)
        print("CRITICAL: Please ensure you run this script from the project root directory 'confidential_computing_summit'.", file=sys.stderr)
        sys.exit(1)

# It's good practice for the main orchestrator to set up basic logging.
# Individual scripts will also have their own more detailed logging.
PIPELINE_LOG_DIR = config.LOG_DIR # Uses LOG_DIR from config.py (to be defined there shortly)
PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log directory exists
PIPELINE_LOG_FILE_NAME = f"pipeline_orchestrator_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
PIPELINE_LOG_FILE = PIPELINE_LOG_DIR / PIPELINE_LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO, # Default level, will be updated by --verbose arg parsing later
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(PIPELINE_LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PipelineOrchestrator")


# --- Configuration for Phase 0: Audio Transcription ---
# PHASE0_INPUT_AUDIO_FILENAME is now defined in config.py as config.PHASE0_INPUT_AUDIO_FILENAME
# Output will be config.MASTER_TRANSCRIPT_FILE as defined in scripts/utils/config.py


def run_phase(phase_name: str, module_name: str, day_identifier: str | None = None, verbose: bool = False): # Add verbose parameter
    """
    Runs a given phase of the pipeline as a Python module.
    Passes the day_identifier if provided.
    Passes --verbose if verbose is True.
    """
    logger.info(f"--- Starting {phase_name} for day: {day_identifier if day_identifier else 'N/A (not day-specific or default)'} ---")

    command = [sys.executable, "-m", module_name]
    if day_identifier:
        # Assuming all phase scripts that need a day_id accept it via --day
        # This convention needs to be implemented in those scripts.
        command.extend(["--day", day_identifier])

    try:
        if verbose: # Add this block
            command.append("--verbose")
            logger.debug(f"Attempting to run {module_name} with --verbose flag.")

        logger.debug(f"Executing command: {' '.join(command)} in CWD: {config.PROJECT_ROOT}") # Log the command
        # Using sys.executable to ensure it uses the same Python interpreter
        # as the one running this pipeline script (especially important for venvs).
        result = subprocess.run(
            command,
            # capture_output=True, # Temporarily disabled for testing
            # text=True,           # Temporarily disabled for testing
            check=False,
            cwd=config.PROJECT_ROOT,
            timeout=3600  # Increased timeout to 1 hour (3600 seconds)
        )

        # Log stdout and stderr from the subprocess
        # These blocks will likely not run if capture_output=False, as result.stdout/stderr will be None
        if hasattr(result, 'stdout') and result.stdout:
            logger.info(f"Output from {module_name} (if captured):\n{result.stdout}")
        if hasattr(result, 'stderr') and result.stderr:
            if result.returncode != 0:
                logger.error(f"Errors from {module_name} (if captured):\n{result.stderr}")
            else:
                logger.warning(f"Stderr output (possibly warnings) from {module_name} (if captured):\n{result.stderr}")

        if result.returncode != 0:
            logger.error(f"{phase_name} FAILED with return code {result.returncode}.")
            logger.info(f"Check terminal for direct output from {module_name} as capture_output was likely disabled for this run.")
            return False
        logger.info(f"--- {phase_name} COMPLETED successfully ---")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{phase_name} ({module_name}) TIMED OUT after 3600 seconds.")
        logger.info(f"Check terminal for direct output from {module_name} as capture_output was likely disabled for this run.")
        return False
    except FileNotFoundError:
        logger.error(f"Error: The Python interpreter '{sys.executable}' or module '{module_name}' was not found.")
        logger.error("Ensure Python is in your PATH and the module path is correct.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running {phase_name} ({module_name}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main_pipeline(day_identifier: str, verbose: bool = False): # Add verbose parameter
    logger.info(f"====== STARTING AI CONFERENCE NOTE-TAKING PIPELINE for {day_identifier} ======")
    if verbose: # Add this line
        logger.debug(f"Verbose mode active for pipeline execution.") # Add this line


    # Get day-specific paths using functions from config
    # These are the primary paths that vary per day for inputs/outputs of phases.
    # Phase scripts will internally use these functions with the day_identifier.
    master_transcript_file_for_day = config.get_master_transcript_file(day_identifier)
    final_pdf_file_for_day = config.get_final_pdf_file(day_identifier)

    # Phase 0: Audio Transcription
    # Assumes scripts.phase0_transcribe_audio is refactored/configured to:
    # 1. Accept --day argument.
    # 2. Check for existing output (config.get_master_transcript_file(day_id)) and skip if present & non-empty.
    # 3. Process config.get_raw_audio_file(day_id).
    # 4. Write output to config.get_master_transcript_file(day_id).
    # 5. Return appropriate exit code (0 for success/skipped, non-zero for failure).
    if not run_phase("Phase 0: Audio Transcription", "scripts.phase0_transcribe_audio", day_identifier, verbose=verbose): # Pass verbose
        logger.error(f"Audio transcription (Phase 0) for {day_identifier} failed. Halting pipeline.")
        sys.exit(1)

    # Explicitly check output of Phase 0
    if not master_transcript_file_for_day.exists() or master_transcript_file_for_day.stat().st_size == 0:
        logger.error(f"Phase 0 for {day_identifier} was run, but master transcript file {master_transcript_file_for_day} is missing or empty.")
        logger.error("This might be because the transcription script skipped (if file existed and was non-empty) or failed to produce output.")
        logger.error(f"Halting pipeline. To re-run transcription for {day_identifier}, delete the existing transcript file if it's incorrect or incomplete.")
        sys.exit(1)
    logger.info(f"Phase 0 output {master_transcript_file_for_day} confirmed for {day_identifier}.")

    # Phase 1: Asset Pre-processing & Extraction
    # Assumes scripts.1_preprocess_slides accepts --day and uses config.get_raw_slides_dir(day_id)
    if not run_phase("Phase 1: Asset Pre-processing & Extraction", "scripts.phase1_asset_processing", day_identifier, verbose=verbose): # Pass verbose
        sys.exit(1)

    # Phase 2: Contextualization & Mapping
    # Assumes scripts.2_contextualize_map accepts --day and uses relevant config.get_..._file(day_id)
    if not run_phase("Phase 2: Contextualization & Mapping", "scripts.phase2_contextualization", day_identifier, verbose=verbose): # Pass verbose
        sys.exit(1)

    # Phase 3: Synthesis & Enrichment
    # Assumes scripts.phase3_synthesis accepts --day and uses relevant config.get_..._file(day_id)
    if not run_phase("Phase 3: Synthesis & Enrichment", "scripts.phase3_synthesis", day_identifier, verbose=verbose): # Pass verbose
        sys.exit(1)

    # Phase 4: Final Compilation & Delivery
    # Assumes scripts.phase4_compilation accepts --day and uses relevant config.get_final_pdf_file(day_id)
    if not run_phase("Phase 4: Final Compilation & Delivery", "scripts.phase4_compilation", day_identifier, verbose=verbose): # Pass verbose
        sys.exit(1)

    logger.info(f"====== AI CONFERENCE NOTE-TAKING PIPELINE FOR {day_identifier} COMPLETED SUCCESSFULLY ======")
    logger.info(f"Final PDF notes should be available at: {final_pdf_file_for_day}")
    logger.info(f"Pipeline orchestrator log: {PIPELINE_LOG_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Conference Note-Taking Pipeline for a specific day.")
    parser.add_argument(
        "--day",
        type=str,
        required=True,
        help="Day identifier (e.g., 'day_1', 'day_2'). This will be used to find input files like 'materials/<day>_notes.txt', 'materials/<day>_slides/', etc."
    )
    parser.add_argument(
        "--verbose",
        action="store_true", # Add this line
        help="Enable verbose logging for the orchestrator and pass --verbose to sub-scripts." # Add this line
    )
    args = parser.parse_args()

    # Configure logging level based on --verbose flag (add these lines)
    if args.verbose:
        # Set level for the logger instance used in this script
        logger.setLevel(logging.DEBUG)
        # Also ensure all handlers attached to the root logger (which basicConfig configures)
        # are also set to DEBUG, otherwise they might filter out DEBUG messages.
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled for orchestrator.")

    main_pipeline(args.day, verbose=args.verbose) # Pass verbose to main_pipeline
