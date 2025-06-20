import logging
from datetime import datetime
from pathlib import Path
import json
import os
import re
import argparse # For command-line arguments
import sys # For StreamHandler

# Attempt to import necessary modules for LLM interaction
try:
    from dotenv import load_dotenv
except ImportError:
    logging.warning("python-dotenv not installed. .env file may not be loaded automatically by this script.")
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    logging.warning("google-generativeai not installed. LLM functionality will be disabled in Phase 3.")
    genai = None

# Assuming config.py is in a utils subdirectory relative to where this script might be run from
# or that the Python path is set up correctly if run as a module.
try:
    from scripts.utils import config
    from scripts.utils.prompts import PHASE3_SYNTHESIS_PROMPT
except ImportError:
    # Fallback for direct execution if utils is not in path, not ideal
    # This assumes a specific directory structure if run directly.
    # It's better to run scripts as modules from the project root.
    try:
        import config # if config.py is in the same directory
    except ImportError:
        logging.error("CRITICAL: Could not import config.py. Ensure it's accessible.")
        # Define critical fallbacks or exit
        class PlaceholderConfig:
            USER_NOTES_FILE = Path("../materials/day_1_notes.txt") # Adjusted for script in scripts/
            MAPPED_SLIDE_CONTENT_FILE = Path("../working_data/mapped_extracted_slide_content.txt")
            RAW_LLM_AUDIO_SEGMENTATION_OUTPUT_TXT = Path("../working_data/llm_raw_audio_segmentation_output.txt")
            ENRICHED_NOTES_MD_FILE = Path("../working_data/enriched_notes.md")
            TODO_RAW_JSON_FILE = Path("../working_data/todo_raw.json") # Should be get_todo_raw_json_file
            LOG_DIR = Path("../logs") # Fallback LOG_DIR
            UNIVERSAL_GEMINI_MODEL_NAME = "gemini-1.5-pro-preview-0409" # Default if config fails
            # PHASE3_SYNTHESIS_PROMPT fallback is removed as it's imported directly
            PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes this script is in scripts/

        config = PlaceholderConfig()
        logging.warning("Using placeholder config due to import error.")


logger = logging.getLogger(__name__)

def setup_logging(day_identifier: str, verbose: bool = False):
    """Sets up logging for this script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_directory = config.LOG_DIR # Use the central log directory

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_name = f"phase3_synthesis_{day_identifier}_{ts}.log"
    logfile = log_directory / logfile_name

    try:
        log_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL: Could not create log directory {log_directory}: {e}. Logging to file will fail.")
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        logger.error(f"Logging to file disabled due to directory creation error: {e}")
        return

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info("Logging initialised for Phase 3 (Synthesis & Enrichment) -> %s", logfile)

def load_file_content(file_path: Path, description: str) -> str | None:
    """Loads content from a text file, logs errors."""
    if not file_path.is_file():
        logger.error(f"Input file for {description} not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"Successfully loaded {description} from {file_path} (length: {len(content)} chars)")
            if not content.strip() and description != "User manual notes": # User notes can be short
                 logger.warning(f"{description} file {file_path} is empty or contains only whitespace.")
            return content
    except Exception as e:
        logger.error(f"Error reading {description} file {file_path}: {e}")
        return None

def synthesize_notes_with_llm(
    day_identifier: str, # Added day_identifier
    user_manual_notes_str: str,
    mapped_slide_content_str: str,
    segmented_audio_transcripts_str: str
) -> tuple[str | None, list | None]:
    """
    Uses an LLM to synthesize enriched notes based on manual notes, slide content, and audio segments.
    Returns a tuple: (enriched_markdown_notes_string, list_of_todo_items).
    """
    if not genai:
        logger.error("Gemini SDK (google-generativeai) is not available. Cannot perform LLM synthesis.")
        return None, None

    if load_dotenv:
        env_path = getattr(config, 'PROJECT_ROOT', Path(".")) / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env file from {env_path} for Phase 3 LLM call.")
        # No warning if .env not found here, as API key presence is checked next

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables for Phase 3 LLM call. Synthesis aborted.")
        return None, None

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK for Phase 3: {e}")
        return None, None

    prompt = PHASE3_SYNTHESIS_PROMPT.format(
        user_manual_notes_string=user_manual_notes_str,
        mapped_slide_content_string=mapped_slide_content_str,
        segmented_audio_transcripts_string=segmented_audio_transcripts_str
    )

    model_name = getattr(config, 'UNIVERSAL_GEMINI_MODEL_NAME', "gemini-1.5-pro-preview-0409") # Fallback if not in config
    logger.info(f"Sending request to LLM ({model_name}) for note synthesis...")
    # logger.debug(f"Phase 3 Synthesis LLM Prompt Snippet:\n{prompt[:1000]}...")

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        raw_response_text = ""
        if hasattr(response, 'text') and response.text:
            raw_response_text = response.text
        elif hasattr(response, 'parts') and response.parts: # Check for parts if text attribute is empty
            raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        logger.info(f"LLM Raw Response Snippet (Phase 3 Synthesis - first 200 chars): '{raw_response_text[:200]}'")
        logger.debug(f"Full Raw LLM response for Phase 3 Synthesis:\n{raw_response_text}")

        # Save the raw response text before attempting to parse it
        raw_llm_output_file = config.get_raw_llm_phase3_synthesis_output_txt(day_identifier)
        try:
            raw_llm_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_llm_output_file, "w", encoding="utf-8") as f_raw:
                f_raw.write(raw_response_text)
            logger.info(f"Saved raw LLM Phase 3 synthesis response to: {raw_llm_output_file}")
        except Exception as e_save_raw:
            logger.error(f"Could not save raw LLM Phase 3 synthesis response to {raw_llm_output_file}: {e_save_raw}")

        # Parse raw_response_text using new delimiters
        notes_start_delim = "ENRICHED_MARKDOWN_NOTES_START"
        notes_end_delim = "ENRICHED_MARKDOWN_NOTES_END"
        todo_start_delim = "TODO_ITEMS_START"
        todo_end_delim = "TODO_ITEMS_END"

        enriched_notes_md = None
        todo_items_list = [] # Default to empty list

        # Extract enriched_markdown_notes
        notes_start_index = raw_response_text.find(notes_start_delim)
        notes_end_index = raw_response_text.find(notes_end_delim)
        if notes_start_index != -1 and notes_end_index != -1 and notes_end_index > notes_start_index:
            enriched_notes_md = raw_response_text[notes_start_index + len(notes_start_delim) : notes_end_index].strip()
            logger.info("Successfully extracted enriched_markdown_notes block.")
        else:
            logger.warning(f"Could not find '{notes_start_delim}' and '{notes_end_delim}' delimiters in LLM response.")
            # Fallback or error handling if notes are critical
            enriched_notes_md = "" # Default to empty string if not found

        # Extract todo_items
        todo_start_index = raw_response_text.find(todo_start_delim)
        todo_end_index = raw_response_text.find(todo_end_delim)
        if todo_start_index != -1 and todo_end_index != -1 and todo_end_index > todo_start_index:
            todo_block_str = raw_response_text[todo_start_index + len(todo_start_delim) : todo_end_index].strip()
            if todo_block_str: # Ensure it's not empty before splitting
                todo_items_list = [line.strip() for line in todo_block_str.splitlines() if line.strip()]
            logger.info(f"Successfully extracted and parsed todo_items block. Found {len(todo_items_list)} items.")
        else:
            logger.warning(f"Could not find '{todo_start_delim}' and '{todo_end_delim}' delimiters in LLM response. To-do list will be empty.")

        if not enriched_notes_md and not todo_items_list: # If neither block was successfully extracted
             logger.error("Neither enriched notes nor to-do items could be meaningfully extracted from LLM response using delimiters.")
             # Depending on strictness, you might return None, None or ("#Error: No content", [])
             return "# Error: LLM did not produce parsable content for notes.", []


        return enriched_notes_md, todo_items_list

    except Exception as e:
        logger.error(f"Error during LLM call or processing response for Phase 3: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def save_output(content: str, output_path: Path, description: str):
    """Saves string content to a file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote {description} to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing {description} to {output_path}: {e}")

def save_json_output(data: list | dict, output_path: Path, description: str):
    """Saves Python list/dict to a JSON file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully wrote {description} (JSON) to: {output_path}")
    except Exception as e:
        logger.error(f"Error writing {description} (JSON) to {output_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Synthesis & Enrichment for a specific day.")
    parser.add_argument("--day", type=str, required=True, help="Day identifier (e.g., 'day_1', 'day_2')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    day_identifier = args.day

    setup_logging(day_identifier=day_identifier, verbose=args.verbose)

    logger.info(f"=== Phase 3: Synthesis & Enrichment started for {day_identifier} ===")
    if args.verbose:
        logger.debug(f"Verbose logging enabled for Phase 3 script (day: {day_identifier}).")

    # 1. Load input files using day-specific paths
    user_notes_file_path = config.get_user_notes_file(day_identifier)
    mapped_slides_file_path = config.get_mapped_slide_content_file(day_identifier)
    segmented_audio_file_path = config.get_raw_llm_audio_segmentation_output_txt(day_identifier)

    user_notes_str = load_file_content(user_notes_file_path, "User manual notes")
    mapped_slides_str = load_file_content(mapped_slides_file_path, "Mapped slide content")
    segmented_audio_str = load_file_content(segmented_audio_file_path, "LLM-segmented audio transcripts")

    if user_notes_str is None: # Manual notes are essential
        logger.error(f"User manual notes could not be loaded from {user_notes_file_path}. Aborting Phase 3 for {day_identifier}.")
        sys.exit(1)

    # Mapped slides and segmented audio can be empty/None if previous phases had issues;
    # The LLM prompt is designed to handle potentially missing context for these.
    # We'll pass them as empty strings if None, so .format() doesn't fail.
    if mapped_slides_str is None:
        logger.warning(f"{mapped_slides_file_path.name} not loaded. Synthesis will proceed without slide context for {day_identifier}.")
        mapped_slides_str = "# Mapped slide content not available or file empty.\n"

    if segmented_audio_str is None:
        logger.warning(f"{segmented_audio_file_path.name} not loaded. Synthesis will proceed without audio context for {day_identifier}.")
        segmented_audio_str = "# Segmented audio transcripts not available or file empty.\n"
        # CRITICAL: Check if this file was empty from your attachment. If so, this is expected.
        # If it's still empty after Phase 2 should have populated it, then that's an issue for Phase 2.
        if segmented_audio_file_path.exists() and segmented_audio_file_path.stat().st_size == 0:
            logger.warning(f"File {segmented_audio_file_path.name} exists but is empty. This might affect audio placeholder resolution for {day_identifier}.")


    # 2. Call LLM for synthesis
    enriched_notes_md, todo_items_list = synthesize_notes_with_llm(
        day_identifier, # Pass day_identifier
        user_notes_str,
        mapped_slides_str,
        segmented_audio_str
    )

    # Define output file paths using day_identifier
    enriched_notes_output_path = config.get_enriched_notes_md_file(day_identifier)
    todo_json_output_path = config.get_todo_raw_json_file(day_identifier)

    # 3. Save outputs
    if enriched_notes_md is not None:
        save_output(enriched_notes_md, enriched_notes_output_path, "Enriched Markdown notes")
    else:
        logger.error(f"LLM synthesis did not return enriched notes content for {day_identifier}. Output file will not be created or will be empty.")
        # Create an empty/error placeholder if desired
        save_output("# Error: LLM did not produce enriched notes content.\n", enriched_notes_output_path, "Enriched Markdown notes (Error)")


    if todo_items_list is not None: # Can be an empty list, which is valid
        save_json_output(todo_items_list, todo_json_output_path, "To-Do items list")
    else:
        logger.error(f"LLM synthesis did not return a to-do items list for {day_identifier}. To-do JSON file will not be created.")
        # Create an empty/error placeholder if desired
        save_json_output([], todo_json_output_path, "To-Do items list (Error - no data from LLM)")

    logger.info(f"=== Phase 3 for {day_identifier} completed ===")

if __name__ == "__main__":
    main()
