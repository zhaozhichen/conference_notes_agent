import logging
from datetime import datetime
from pathlib import Path
import json
import os # Import the os module
import re
import sys
import argparse # For command-line arguments

# Attempt to import necessary modules for LLM interaction
try:
    from dotenv import load_dotenv
except ImportError:
    logging.warning("python-dotenv not installed. .env file may not be loaded automatically.")
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    logging.warning("google-generativeai not installed. LLM functionality will be disabled.")
    genai = None

from scripts.utils import agenda_utils, transcript_utils, config # Assuming config.py is up-to-date
from scripts.utils.prompts import PHASE2_CONTEXT_PROMPT, PHASE2_MERGE_PROMPT, PHASE2_AUDIO_SEGMENTATION_PROMPT

# Global logger for this module
# Get a logger instance for this script
logger = logging.getLogger(__name__)

def setup_logging(day_identifier: str, verbose: bool = False):
    """Sets up logging for this script, appending to a file and printing to stdout."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_directory = config.LOG_DIR # Use the central log directory

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_name = f"phase2_contextualization_{day_identifier}_{ts}.log"
    logfile = log_directory / logfile_name

    try:
        log_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"CRITICAL: Could not create log directory {log_directory}: {e}. Logging to file will fail.")
        # Fallback: Log only to console if directory creation fails
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
            logging.StreamHandler(sys.stdout) # Also log to console
        ]
    )
    logger.info("Logging initialised for Phase 2 (Contextualization & Mapping) -> %s", logfile)


def load_file_content(file_path: Path, description: str) -> str | None:
    """Loads content from a text file, logs errors."""
    if not file_path.is_file():
        logger.error(f"{description} file not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {description} file {file_path}: {e}")
        return None

def call_llm_for_contextualization(
    day_identifier: str, # Added day_identifier
    agenda_str: str,
    user_notes_str: str,
    extracted_slides_str: str
) -> dict | None:
    """
    Formats a prompt with agenda, user notes, and slide content,
    then calls the configured Gemini LLM to get contextualization. Also saves mapping file.
    Parses the expected JSON response.
    """
    if not genai:
        logger.error("Gemini SDK (google-generativeai) is not available. Cannot call LLM.")
        return None

    # Load API Key - dotenv should be called once at the start of the application
    # For modularity, let's ensure it's loaded here if not already.
    if load_dotenv:
        env_path = config.PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env file from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}. GOOGLE_API_KEY might not be set.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables. LLM call aborted.")
        return None

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK: {e}")
        return None

    prompt = PHASE2_CONTEXT_PROMPT.format(
        agenda_string=agenda_str,
        user_notes_string=user_notes_str,
        extracted_slides_string=extracted_slides_str
    )

    logger.info(f"Sending request to LLM ({config.PHASE2_LLM_MODEL}) for contextualization...")
    # logger.debug(f"LLM Prompt for Phase 2:\n{prompt[:1000]}...") # Log a snippet of the prompt

    try:
        model = genai.GenerativeModel(config.PHASE2_LLM_MODEL)
        response = model.generate_content(prompt)

        # Debug: Log raw response text
        raw_response_text = ""
        if hasattr(response, 'text'):
            raw_response_text = response.text
        elif response.parts:
            raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        logger.debug(f"Raw LLM response text:\n{raw_response_text}")

        # Attempt to find JSON block within the response
        # LLMs sometimes wrap JSON in backticks or add explanations.
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # If no markdown block, assume the whole response is JSON (or try to find a JSON object directly)
            json_str = raw_response_text
            # A more robust way if it's not perfectly clean JSON:
            first_brace = json_str.find('{')
            last_brace = json_str.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = json_str[first_brace : last_brace+1]
            else:
                logger.error("No clear JSON object found in LLM response.")
                return None

        llm_output = json.loads(json_str)
        logger.info("Successfully parsed JSON response from LLM.")

        # Validate expected keys
        if "attended_sessions" not in llm_output or "slide_to_session_mapping" not in llm_output:
            logger.error("LLM response is missing one or more required keys ('attended_sessions', 'slide_to_session_mapping').")
            logger.debug(f"LLM output received: {llm_output}")
            return None

        # Save the slide_to_session_mapping part of the response
        slide_to_session_map_from_llm = llm_output.get("slide_to_session_mapping", [])
        # Ensure config.get_slide_session_mapping_file is available and day_identifier is passed
        slide_session_mapping_file = config.get_slide_session_mapping_file(day_identifier)
        if slide_to_session_map_from_llm: # Only save if not empty
            try:
                slide_session_mapping_file.parent.mkdir(parents=True, exist_ok=True)
                with open(slide_session_mapping_file, "w", encoding="utf-8") as f_map:
                    json.dump(slide_to_session_map_from_llm, f_map, indent=2)
                logger.info(f"LLM response for slide_to_session_mapping saved to: {slide_session_mapping_file}")
            except Exception as e_map_save:
                logger.error(f"Could not save LLM slide-to_session_mapping to {slide_session_mapping_file}: {e_map_save}")
        else:
            logger.warning(f"LLM did not provide slide_to_session_mapping for {day_identifier}, so no mapping file saved.")

        return llm_output
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        logger.error(f"Problematic JSON string was: {json_str}")
        return None
    except Exception as e:
        logger.error(f"Error during LLM call or processing response: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def merge_slide_content_with_llm(day_identifier: str): # Added day_identifier
    """
    Uses an LLM to merge session titles from slide_session_mapping.json
    into extracted_slide_content.txt, saving to mapped_extracted_slide_content.txt.
    Relies on config for file paths and LLM model/prompt.
    """
    logger.info(f"Starting LLM-based merge of session titles into slide content for {day_identifier}.")

    if not genai:
        logger.error("Gemini SDK (google-generativeai) is not available. Cannot perform LLM merge.")
        return

    # Get day-specific paths
    extracted_content_path = config.get_extracted_slide_content_file(day_identifier)
    mapping_path = config.get_slide_session_mapping_file(day_identifier)
    output_path = config.get_mapped_slide_content_file(day_identifier)

    # 1. Load content of extracted_slide_content.txt
    extracted_slides_str = load_file_content(extracted_content_path, "Extracted slide content for merge")
    if extracted_slides_str is None:
        logger.error(f"Cannot perform LLM merge for {day_identifier}: Failed to load {extracted_content_path}.")
        # Create an empty/error MAPPED_SLIDE_CONTENT_FILE
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f_err:
                f_err.write(f"# Error: Source file {extracted_content_path.name} not found or unreadable for LLM merge.\\n")
            logger.info(f"Wrote error placeholder to {output_path}")
        except Exception as e_write_err:
            logger.error(f"Could not write error placeholder to {output_path}: {e_write_err}")
        return

    # 2. Load content of slide_session_mapping.json
    slide_session_mapping_json_str = load_file_content(mapping_path, "Slide-to-session mapping JSON for merge")
    if slide_session_mapping_json_str is None:
        logger.error(f"Cannot perform LLM merge for {day_identifier}: Failed to load {mapping_path}.")
        # Create an empty/error MAPPED_SLIDE_CONTENT_FILE (or copy original if it exists)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f_err:
                f_err.write(extracted_slides_str) # Write original content
                f_err.write(f"\\n# Error: Mapping file {mapping_path.name} not found or unreadable for LLM merge.\\n")
            logger.info(f"Wrote original content and error placeholder to {output_path}")
        except Exception as e_write_err:
            logger.error(f"Could not write to {output_path}: {e_write_err}")
        return

    try:
        json.loads(slide_session_mapping_json_str) # Validate JSON
    except json.JSONDecodeError as e:
        logger.error(f"Cannot perform LLM merge for {day_identifier}: Content of {mapping_path} is not valid JSON: {e}")
        # Similar fallback for writing to MAPPED_SLIDE_CONTENT_FILE
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f_err:
                f_err.write(extracted_slides_str)
                f_err.write(f"\\n# Error: Mapping file {mapping_path.name} content is not valid JSON: {e}\\n")
            logger.info(f"Wrote original content and JSON error placeholder to {output_path}")
        except Exception as e_write_err:
            logger.error(f"Could not write to {output_path}: {e_write_err}")
        return

    # Load API Key
    if load_dotenv: # Ensure dotenv is loaded (should be done once at script start ideally)
        env_path = config.PROJECT_ROOT / ".env"
        if env_path.exists(): load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found for LLM merge. Aborting.")
        return
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK for LLM merge: {e}")
        return

    # 3. Format the prompt
    prompt_for_merge = PHASE2_MERGE_PROMPT.format(
        extracted_slides_string=extracted_slides_str,
        slide_session_mapping_json_string=slide_session_mapping_json_str
    )

    logger.info(f"Sending request to LLM ({config.PHASE2_LLM_MODEL}) for merging session titles...")
    # logger.debug(f"LLM Merge Prompt Snippet:\\\\n{prompt_for_merge[:500]}...\") # Log snippet

    try:
        model = genai.GenerativeModel(config.UNIVERSAL_GEMINI_MODEL_NAME) # Use universal model
        response = model.generate_content(prompt_for_merge)

        merged_content_text = ""
        if hasattr(response, 'text') and response.text:
            merged_content_text = response.text
        elif response.parts: # Handle cases where response might be in parts
            merged_content_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        if not merged_content_text.strip():
            logger.warning("LLM returned empty or whitespace-only content for the merge operation.")
            merged_content_text = f"# LLM returned no content for merge operation.\\nOriginal extracted content was:\\n{extracted_slides_str}"

        # 4. Write the LLM's response to mapped_extracted_slide_content.txt
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_content_text)
        logger.info(f"LLM-merged slide content successfully written to: {output_path}")

    except Exception as e:
        logger.error(f"Error during LLM call or writing merged content: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f_err:
                f_err.write(f"# Error during LLM merge operation: {e}\\nOriginal extracted content was:\\n{extracted_slides_str}") # extracted_slides_str should be defined in this scope
            logger.info(f"Wrote error placeholder and original content to {output_path}")
        except Exception as e_write_err:
            logger.error(f"Could not even write error placeholder to {output_path}: {e_write_err}")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Contextualization and Mapping for a specific day.")
    parser.add_argument("--day", type=str, required=True, help="Day identifier (e.g., 'day_1', 'day_2')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    day_identifier = args.day

    setup_logging(day_identifier=day_identifier, verbose=args.verbose)

    logger.info(f"=== Phase 2: Contextualization & Mapping (LLM-based) started for {day_identifier} ===")
    if args.verbose:
        logger.debug(f"Verbose logging enabled for Phase 2 script (day: {day_identifier}).")

    # 1. Load all necessary text inputs
    logger.info(f"Loading agenda from: {config.AGENDA_FILE}") # AGENDA_FILE is not day-specific
    parsed_agenda_sessions = agenda_utils.parse_local_agenda(config.AGENDA_FILE)
    if not parsed_agenda_sessions:
        logger.error("Failed to parse agenda. Aborting Phase 2.")
        sys.exit(1)

    # Convert parsed agenda to a simple string format for the LLM
    agenda_string_for_llm = "\n".join([f"Title: {s['title']}, Day: {s.get('day', 'N/A')}, Start: {s['start']}, End: {s.get('end', 'N/A')}" for s in parsed_agenda_sessions])

    user_notes_file = config.get_user_notes_file(day_identifier)
    logger.info(f"Loading user notes from: {user_notes_file}")
    user_notes_content_str = load_file_content(user_notes_file, "User notes")
    if user_notes_content_str is None:
        logger.error("Failed to load user notes. Aborting Phase 2.")
        sys.exit(1)

    extracted_slide_content_file = config.get_extracted_slide_content_file(day_identifier)
    logger.info(f"Loading extracted slide content from: {extracted_slide_content_file}")
    extracted_slides_content_str = load_file_content(extracted_slide_content_file, "Extracted slide content")
    if extracted_slides_content_str is None:
        logger.error("Failed to load extracted slide content. Aborting Phase 2.")
        sys.exit(1)

    # 2. Call LLM for contextualization
    llm_context_data = call_llm_for_contextualization(
        day_identifier, # Pass day_identifier
        agenda_string_for_llm,
        user_notes_content_str,
        extracted_slides_content_str
    )

    if not llm_context_data:
        logger.error("Failed to get contextualization data from LLM. Aborting Phase 2.")
        sys.exit(1)

    attended_session_titles_from_llm = llm_context_data.get("attended_sessions", [])
    slide_to_session_map_from_llm = llm_context_data.get("slide_to_session_mapping", [])

    if not attended_session_titles_from_llm:
        logger.warning("LLM did not identify any attended sessions.")
    else:
        logger.info(f"LLM identified {len(attended_session_titles_from_llm)} attended sessions: {attended_session_titles_from_llm}")

    if not slide_to_session_map_from_llm:
        logger.warning("LLM did not provide any slide-to-session mappings.")
    else:
        logger.info(f"LLM provided {len(slide_to_session_map_from_llm)} slide-to-session mappings.")

    # 3. Use LLM to merge session titles into the extracted slide content file
    # This step relies on the existence of EXTRACTED_SLIDE_CONTENT_FILE (from Phase 1)
    # and SLIDE_SESSION_MAPPING_FILE (from the first LLM call in this Phase 2 script).
    # Get day-specific paths for these files
    current_extracted_content_file = config.get_extracted_slide_content_file(day_identifier)
    current_slide_mapping_file = config.get_slide_session_mapping_file(day_identifier)
    current_mapped_content_file = config.get_mapped_slide_content_file(day_identifier)

    logger.info(f"Checking for input files for LLM merge: {current_extracted_content_file.name} and {current_slide_mapping_file.name}")

    # We proceed if the first LLM call was successful (implies slide_to_session_map_from_llm has data AND was saved)
    # and if the extracted_slides_content_str was loaded.
    if extracted_slides_content_str is not None and slide_to_session_map_from_llm: # Check if map has content from first LLM
        if current_extracted_content_file.is_file() and current_slide_mapping_file.is_file():
            logger.info("Both input files for merge found. Proceeding with LLM merge for session titles.")
            merge_slide_content_with_llm(day_identifier) # Pass day_identifier
        else:
            # This case should ideally not be hit if slide_to_session_map_from_llm has content,
            # as it implies SLIDE_SESSION_MAPPING_FILE should have been saved.
            # And extracted_slides_content_str being not None implies EXTRACTED_SLIDE_CONTENT_FILE was read.
            if not current_extracted_content_file.is_file():
                logger.error(f"Input file for merge missing: {current_extracted_content_file}. Cannot merge session titles.")
            if not current_slide_mapping_file.is_file():
                logger.error(f"Input file for merge missing: {current_slide_mapping_file}. Cannot merge session titles.")
            # Fallback: Copy original extracted content if it exists, or write placeholder.
            try:
                current_mapped_content_file.parent.mkdir(parents=True, exist_ok=True)
                if extracted_slides_content_str:
                    with open(current_mapped_content_file, "w", encoding="utf-8") as f_out:
                        f_out.write(extracted_slides_content_str)
                    logger.info(f"Copied original slide content to {current_mapped_content_file} as merge inputs were incomplete.")
                else:
                    with open(current_mapped_content_file, "w", encoding="utf-8") as f_empty:
                        f_empty.write("# Merge operation skipped. Original extracted content was empty or one of the required input files for merge was missing.\n")
                    logger.info(f"Wrote placeholder to {current_mapped_content_file} as merge inputs were incomplete or original was empty.")
            except Exception as e_copy_fallback:
                logger.error(f"Could not copy original/placeholder content to {current_mapped_content_file}: {e_copy_fallback}")

    elif not extracted_slides_content_str:
         logger.warning("Original extracted_slide_content.txt was empty or not loaded. Skipping LLM merge step.")
         # Create empty mapped file
         try:
            current_mapped_content_file.parent.mkdir(parents=True, exist_ok=True)
            with open(current_mapped_content_file, "w", encoding="utf-8") as f_empty:
                f_empty.write("# Original extracted_slide_content.txt was empty or not loaded. Merge skipped.\n")
            logger.info(f"Wrote placeholder to {current_mapped_content_file} as original was empty.")
         except Exception as e_write_placeholder_empty:
             logger.error(f"Could not write placeholder to {current_mapped_content_file}: {e_write_placeholder_empty}")

    elif not slide_to_session_map_from_llm: # This means first LLM call failed to produce a map.
        logger.warning("LLM provided no slide mappings from initial contextualization. Session titles cannot be merged using LLM.")
        # Fallback: Copy original extracted content if it exists.
        try:
            if extracted_slides_content_str: # only if original content exists
                current_mapped_content_file.parent.mkdir(parents=True, exist_ok=True)
                with open(current_mapped_content_file, "w", encoding="utf-8") as f_out:
                    f_out.write(extracted_slides_content_str)
                logger.info(f"Copied original slide content to {current_mapped_content_file} as LLM mapping for titles was empty.")
            else: # Original was also empty
                current_mapped_content_file.parent.mkdir(parents=True, exist_ok=True)
                with open(current_mapped_content_file, "w", encoding="utf-8") as f_empty:
                    f_empty.write("# Merge operation skipped. Original extracted content was empty AND LLM mapping was empty.\\n")
                logger.info(f"Wrote placeholder to {current_mapped_content_file} as inputs were empty.")
        except Exception as e_copy_fallback_no_map:
            logger.error(f"Could not copy original slide content to {current_mapped_content_file}: {e_copy_fallback_no_map}")

    # 4. Segment master transcript using LLM based on identified attended sessions
    logger.info("--- Starting LLM-based Audio Transcript Segmentation (Raw Text Output) ---")
    master_transcript_file = config.get_master_transcript_file(day_identifier)
    master_transcript_content_str = load_file_content(master_transcript_file, "Master audio transcript")

    if master_transcript_content_str is None:
        logger.error(f"Master transcript file {master_transcript_file} not loaded. Skipping audio segmentation.")
    elif not attended_session_titles_from_llm:
        logger.warning("No attended session titles identified by LLM. Skipping audio segmentation.")
    else:
        segment_transcript_with_llm( # Call updated function
            day_identifier, # Pass day_identifier
            master_transcript_str=master_transcript_content_str,
            agenda_str=agenda_string_for_llm, # Re-use the stringified agenda
            attended_session_titles_list=attended_session_titles_from_llm
            # parsed_agenda_sessions_list is no longer needed by the simplified function
        )

    logger.info("--- Checking state before saving slide_session_mapping.json ---")
    # 5. (Optional) Save the raw LLM slide-to-session mapping if needed for audit/debug
    # This was defined in execution_plan as SLIDE_SESSION_MAPPING_FILE
    logger.debug(f"Before attempting to save slide_to_session_map_from_llm. "
                 f"Type: {type(slide_to_session_map_from_llm)}, "
                 f"Is None: {slide_to_session_map_from_llm is None}, "
                 f"Length: {len(slide_to_session_map_from_llm) if hasattr(slide_to_session_map_from_llm, '__len__') else 'N/A'}.")
    # current_slide_mapping_file is already defined earlier in main()
    if slide_to_session_map_from_llm: # This is the list of mappings from the LLM
        logger.debug("Condition 'if slide_to_session_map_from_llm:' was TRUE. Proceeding to save.")
        try:
            logger.debug(f"Ensuring parent directory exists for: {current_slide_mapping_file}")
            current_slide_mapping_file.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Attempting to dump JSON to {current_slide_mapping_file}")
            with open(current_slide_mapping_file, "w", encoding="utf-8") as f_map:
                json.dump(slide_to_session_map_from_llm, f_map, indent=2)
            logger.info(f"LLM's slide-to-session mapping saved to: {current_slide_mapping_file}")
        except Exception as e_map_save:
            logger.error(f"Could not save LLM slide-to-session mapping to {current_slide_mapping_file}: {e_map_save}")
            import traceback
            logger.error(f"Traceback for saving error: {traceback.format_exc()}")
    else:
        logger.warning("Skipping save of slide_to_session_map_from_llm because it is empty or None.")

    logger.info(f"=== Phase 2: Contextualization & Mapping for {day_identifier} completed successfully ===")
# sanitize_filename is no longer needed as we are not creating individual session files from titles.
# def sanitize_filename(title: str) -> str:
#     """Sanitizes a session title to be a valid filename component."""
#     title = re.sub(r'[\\/*?:"<>|]', "", title)
#     title = title.replace(" ", "_").replace("-", "_").lower()
#     max_len = 50
#     if len(title) > max_len:
#         title = title[:max_len]
#     return title

def segment_transcript_with_llm(
    day_identifier: str, # Added day_identifier
    master_transcript_str: str,
    agenda_str: str,
    attended_session_titles_list: list
):
    """
    Uses an LLM to segment the master transcript based on attended session titles and agenda.
    Saves the LLM's raw text response (which should be the segmented transcript with titles)
    directly to a file specified in config.RAW_LLM_AUDIO_SEGMENTATION_OUTPUT_TXT.
    """
    logger.info(f"Initiating LLM call for audio transcript segmentation for {day_identifier} (raw text output).")

    # Define output_file_path at the start of the function scope to ensure it's available in all try/except blocks
    output_file_path = config.get_raw_llm_audio_segmentation_output_txt(day_identifier)

    if not genai:
        logger.error("Gemini SDK (google-generativeai) is not available for audio segmentation.")
        return

    # Prepare API key
    if load_dotenv:
        env_path = config.PROJECT_ROOT / ".env"
        if env_path.exists(): load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found for transcript segmentation LLM call. Aborting.")
        return
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK for transcript segmentation: {e}")
        return

    attended_session_titles_string = "\\n".join([f"- {title}" for title in attended_session_titles_list])

    prompt = PHASE2_AUDIO_SEGMENTATION_PROMPT.format(
        agenda_string=agenda_str,
        attended_session_titles_string=attended_session_titles_string,
        master_transcript_string=master_transcript_str
    )

    logger.info(f"Sending request to LLM ({config.UNIVERSAL_GEMINI_MODEL_NAME}) for audio segmentation...")
    # logger.debug(f"Audio Segmentation LLM Prompt Snippet:\\n{prompt[:1000]}...")

    try:
        model = genai.GenerativeModel(config.UNIVERSAL_GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)

        raw_response_text = ""
        if hasattr(response, 'text') and response.text:
            raw_response_text = response.text
        elif response.parts:
            raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            logger.warning("LLM response for audio segmentation had no 'text' attribute or 'parts'.")
            raw_response_text = "# LLM response was empty or in an unexpected format."

        logger.info(f"LLM Raw Response Snippet (Audio Segmentation - first 200 chars): '{raw_response_text[:200]}'")
        logger.debug(f"Full Raw LLM response for audio segmentation:\\n{raw_response_text}")

        # Save the raw LLM response directly to a text file
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f_raw_out:
                f_raw_out.write(raw_response_text)
            logger.info(f"Successfully saved raw LLM audio segmentation output to: {output_file_path}")
        except Exception as e_write_raw:
            logger.error(f"Error writing raw LLM audio segmentation output to {output_file_path}: {e_write_raw}")

    except Exception as e: # Catch errors from model.generate_content or other unexpected issues
        logger.error(f"Error during LLM call for audio segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Attempt to write an error message to the output file
        try:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, "w", encoding="utf-8") as f_err_out:
                f_err_out.write(f"# Error during LLM call for audio segmentation: {e}\n{traceback.format_exc()}")
            logger.info(f"Wrote error details to {output_file_path}")
        except Exception as e_final_err:
            logger.error(f"Could not even write error details to {output_file_path}: {e_final_err}")


    logger.info("=== Phase 2 completed ===")

if __name__ == "__main__":
    # This basicConfig is for when the script is run directly.
    # It might be overridden if main() calls setup_logging() which also calls basicConfig.
    # Consider only calling setup_logging() from main() to avoid this.
    # For now, removing this direct basicConfig here to rely on setup_logging in main.
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
