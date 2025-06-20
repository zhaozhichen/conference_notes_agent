from pathlib import Path
# from . import prompts

# Determine the project root directory.
# This assumes 'config.py' is in 'confidential_computing_summit/scripts/utils/'.
# So, Path(__file__).resolve() gives the path to config.py.
# .parent gives 'utils/', .parent.parent gives 'scripts/', .parent.parent.parent gives 'confidential_computing_summit/'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Base Directories ---
MATERIALS_DIR = PROJECT_ROOT / "materials"
WORKING_DATA_DIR = PROJECT_ROOT / "working_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs" # Central directory for all log files

# --- Day-Specific Base Working Directories ---
def get_day_working_data_dir(day_identifier: str) -> Path:
    """e.g., day_1 -> working_data/day_1/"""
    return WORKING_DATA_DIR / day_identifier


# --- Path Generating Functions for Day-Specific RAW Input Files (mostly in materials) ---
def get_raw_slides_dir(day_identifier: str) -> Path:
    """e.g., day_1 -> materials/day_1_slides/"""
    return MATERIALS_DIR / f"{day_identifier}_slides"

def get_user_notes_file(day_identifier: str) -> Path:
    """e.g., day_1 -> materials/day_1_notes.txt"""
    return MATERIALS_DIR / f"{day_identifier}_notes.txt"

def get_master_transcript_file(day_identifier: str) -> Path:
    """e.g., day_1 -> materials/day_1_recording_transcript.txt"""
    return MATERIALS_DIR / f"{day_identifier}_recording_transcript.txt"

def get_raw_audio_files(day_identifier: str) -> list[Path]:
    """
    Finds raw audio files for a given day, supporting multiple parts and common formats.
    Searches first in `materials/{day_identifier}_recordings/`, then falls back to `materials/{day_identifier}_recording*`.
    e.g., day_3 -> [materials/day_3_recordings/day_3_recording_part_01.m4a, ...]
             or -> [materials/day_1_recording.aac]
    Returns a sorted list of Path objects.
    """
    found_files = []
    supported_extensions = ['.m4a', '.aac', '.mp3', '.wav', '.ogg', '.flac', '.opus']

    # Path for the new day-specific recordings subdirectory
    day_recordings_dir = MATERIALS_DIR / f"{day_identifier}_recordings"

    if day_recordings_dir.is_dir():
        # Prefer files from the day-specific recordings subdirectory
        # These files can have any name, but must have a supported extension.
        for p in day_recordings_dir.glob("*"): 
            if p.is_file() and p.suffix.lower() in supported_extensions:
                found_files.append(p)
    
    # If no files were found in the specific subdirectory (or it didn't exist),
    # then fallback to searching in the main materials directory with the original pattern.
    if not found_files:
        # General pattern to find files starting with "{day_identifier}_recording"
        # This will catch:
        # - day_1_recording.aac
        # - day_1_recording.m4a
        # - day_1_recording_part_01.m4a
        # etc.
        base_pattern = f"{day_identifier}_recording*"
        for p in MATERIALS_DIR.glob(base_pattern):
            if p.is_file() and p.suffix.lower() in supported_extensions:
                found_files.append(p)
            
    # Sort files to ensure parts are in order (e.g., _part_01, _part_02)
    # Default sorting of Path objects (alphabetical on full path string) should work.
    return sorted(list(set(found_files))) # set to ensure uniqueness if patterns overlap or if both paths somehow match

# --- Generic (Non-Day-Specific) Input Files ---
AGENDA_FILE = MATERIALS_DIR / "agenda.txt" # Assuming agenda is common for all days or day is handled by parser

# --- Day-Specific Working/Processed Data Path Functions ---

def get_temp_audio_chunks_dir(day_identifier: str) -> Path:
    """e.g., day_1 -> working_data/day_1/temp_audio_chunks/"""
    return get_day_working_data_dir(day_identifier) / "temp_audio_chunks"

def get_extracted_slide_content_file(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "extracted_slide_content.txt"

def get_mapped_slide_content_file(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "mapped_extracted_slide_content.txt"

def get_cropped_slides_dir(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "cropped_slides"

def get_session_transcripts_dir(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "session_transcripts"

def get_raw_llm_audio_segmentation_output_txt(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "llm_raw_audio_segmentation_output.txt"

def get_slide_session_mapping_file(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "slide_session_mapping.json"

def get_enriched_notes_md_file(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "enriched_notes.md"

def get_todo_raw_json_file(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "todo_raw.json"

def get_raw_llm_phase3_synthesis_output_txt(day_identifier: str) -> Path:
    return get_day_working_data_dir(day_identifier) / "llm_raw_phase3_synthesis_output.txt"

# --- Output Path Function ---
def get_final_pdf_file(day_identifier: str) -> Path:
    """e.g., day_1 -> output/Conference_Notes_CCS25_day_1.pdf"""
    return OUTPUT_DIR / f"Conference_Notes_CCS25_{day_identifier}.pdf"

# --- Constants used by old fixed-day logic (can be removed or adapted) ---
# These are now superseded by the functions above if day_identifier is used.
# Keeping them for reference during transition or if a default day is ever needed.
_DEFAULT_DAY_ID = "day_1"
DEFAULT_USER_NOTES_FILE = get_user_notes_file(_DEFAULT_DAY_ID)
DEFAULT_MASTER_TRANSCRIPT_FILE = get_master_transcript_file(_DEFAULT_DAY_ID)
DEFAULT_RAW_SLIDES_DIR = get_raw_slides_dir(_DEFAULT_DAY_ID)
DEFAULT_RAW_AUDIO_FILES = get_raw_audio_files(_DEFAULT_DAY_ID) # Renamed variable and calls new function
DEFAULT_FINAL_PDF_FILE = get_final_pdf_file(_DEFAULT_DAY_ID)

# --- Slide Processing Configuration ---\
# Supported image extensions for slide processing (primarily for finding .jpg for content)
SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg'] # HEIC is handled separately for metadata

# Configuration for Phase 1: Slide Image Preparation (cropping & fullscreen adaptation)
TARGET_IMAGE_WIDTH = 1920
TARGET_IMAGE_HEIGHT = 1080
CANVAS_BACKGROUND_COLOR = (0, 0, 0) # Black (R, G, B) for Pillow

# --- Gemini / Vision Settings ---
# These would ideally be loaded from .env if they vary, or defined here if static.

# --- Universal Gemini Model Configuration ---
# Define a single model name to be used for all Gemini interactions (vision, text, etc.)
# Ensure this model is accessible with your API key and supports the required modalities.
UNIVERSAL_GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-06-05"
# UNIVERSAL_GEMINI_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"

# For GEMINI_VISION_MODEL, ensure it's a valid model name you have access to.
GEMINI_VISION_MODEL = UNIVERSAL_GEMINI_MODEL_NAME # Using the universal model for vision tasks

# --- Phase 2 LLM Contextualization Settings ---
# Model for text-based analysis and reasoning (Phase 2)
PHASE2_LLM_MODEL = UNIVERSAL_GEMINI_MODEL_NAME # Using the universal model for contextualization

# --- Path for Raw LLM Audio Segmentation Output ---
# RAW_LLM_AUDIO_SEGMENTATION_OUTPUT_TXT is now a function get_raw_llm_audio_segmentation_output_txt(day_identifier)


# --- Phase 2 LLM Audio Segmentation Settings ---
# Note: The output of this prompt is now saved directly as text, not parsed as JSON for individual files.
# Prompt definition moved to scripts/utils/prompts.py

# --- Phase 3 LLM Synthesis & Enrichment Settings ---
# Prompt definition moved to scripts/utils/prompts.py

# --- API Keys & Environment Variables ---
# API keys should be loaded from .env using a library like python-dotenv in the main scripts,
# not hardcoded here. Example:
# import os
# from dotenv import load_dotenv
# load_dotenv(PROJECT_ROOT / ".env") # Assuming .env is in the PROJECT_ROOT
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this matches your .env variable name

# --- Other Configurations ---
# Example: Filename for user\'s raw [todo] items list from Phase 3
# TODO_RAW_JSON_FILE is already defined above under WORKING_DATA_DIR paths.

# Ensure necessary directories exist (optional: scripts can create them)
# For example, the main script or a setup script could call this:
def ensure_directories_exist(day_identifier: str = _DEFAULT_DAY_ID):
    # List of directories that need to exist
    dirs_to_create = [
        MATERIALS_DIR,
        LOG_DIR,  # Central log directory
        OUTPUT_DIR,
        get_day_working_data_dir(day_identifier),  # Base day-specific working dir
        get_raw_slides_dir(day_identifier),
        get_temp_audio_chunks_dir(day_identifier),
        get_cropped_slides_dir(day_identifier),
        get_session_transcripts_dir(day_identifier)
    ]

    # Add parent directories of files (if the file path itself isn't a directory)
    # These are typically directories containing output files from various phases.
    files_whose_parents_need_creating = [
        # Raw audio files (now handled by get_raw_audio_files) are located directly in MATERIALS_DIR.
        # MATERIALS_DIR is already included in 'dirs_to_create', so no separate parent check needed here.
        get_master_transcript_file(day_identifier),
        get_final_pdf_file(day_identifier),
        get_extracted_slide_content_file(day_identifier),
        get_mapped_slide_content_file(day_identifier),
        get_raw_llm_audio_segmentation_output_txt(day_identifier),
        get_slide_session_mapping_file(day_identifier),
        get_enriched_notes_md_file(day_identifier),
        get_todo_raw_json_file(day_identifier),
        get_raw_llm_phase3_synthesis_output_txt(day_identifier) # Add new path
    ]

    for file_path_obj in files_whose_parents_need_creating:
        dirs_to_create.append(file_path_obj.parent)

    # Create all collected directories
    # Using a set to avoid attempting to create the same directory multiple times
    for d_path in set(dirs_to_create):
        d_path.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # This can be run to print out the configured paths for verification
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MATERIALS_DIR: {MATERIALS_DIR}")
    print(f"AGENDA_FILE: {AGENDA_FILE}")
    print(f"WORKING_DATA_DIR: {WORKING_DATA_DIR}")
    print(f"CROPPED_SLIDES_DIR: {CROPPED_SLIDES_DIR}")
    print(f"LOG_DIR: {LOG_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"--- Example paths for day_id='{_DEFAULT_DAY_ID}' ---")
    print(f"Raw Slides Dir ({_DEFAULT_DAY_ID}): {get_raw_slides_dir(_DEFAULT_DAY_ID)}")
    print(f"User Notes File ({_DEFAULT_DAY_ID}): {get_user_notes_file(_DEFAULT_DAY_ID)}")
    print(f"Master Transcript File ({_DEFAULT_DAY_ID}): {get_master_transcript_file(_DEFAULT_DAY_ID)}")
    print(f"Raw Audio Files ({_DEFAULT_DAY_ID}): {get_raw_audio_files(_DEFAULT_DAY_ID)}")
    print(f"Final PDF File ({_DEFAULT_DAY_ID}): {get_final_pdf_file(_DEFAULT_DAY_ID)}")
    # ensure_directories_exist(_DEFAULT_DAY_ID) # Optionally run this to create them for the default day
    # print("Checked/created necessary directories.")
