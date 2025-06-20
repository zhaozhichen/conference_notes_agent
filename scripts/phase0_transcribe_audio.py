import logging
from datetime import datetime
from pathlib import Path
import json
import os
import re
import subprocess
import shutil # Import the shutil module
import time
import traceback
import sys
import argparse # Added for command-line arguments

# Attempt to import necessary modules for LLM interaction
try:
    from dotenv import load_dotenv
except ImportError:
    # This script should still function if dotenv is not found,
    # relying on environment variables being set by other means.
    # A warning will be logged in main if GOOGLE_API_KEY is missing.
    load_dotenv = None
    print("Warning: python-dotenv not installed. .env file may not be loaded automatically by phase0_transcribe_audio.py.")


try:
    import google.generativeai as genai
except ImportError:
    # If genai is not installed, the script cannot function.
    # Log this critical error and ensure main() handles it.
    print("CRITICAL ERROR: google-generativeai not installed. Audio transcription cannot proceed.")
    genai = None

# Assuming config.py is in a utils subdirectory
try:
    from .utils import config
except ImportError:
    # This fallback is less ideal but can help if running the script directly out of its intended module context
    # However, the pipeline script (run_pipeline.py) should run this as a module.
    try:
        from utils import config # If running from 'scripts' directory
    except ImportError:
        print("CRITICAL ERROR: Could not import config.py. Ensure it's in scripts/utils/ and project is run correctly.")
        # Define absolutely essential fallbacks or sys.exit()
        class PlaceholderConfig: # Very basic fallback
            DAY1_RAW_AUDIO_FILE = Path("../materials/day_1_recording.aac")
            MASTER_TRANSCRIPT_FILE = Path("../materials/day_1_recording_transcript.txt")
            SCRATCHPAD_DIR = Path("../working_data/scratchpad")
            UNIVERSAL_GEMINI_MODEL_NAME = "gemini-1.5-pro-preview-0409" # Must match your available model
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
            # Chunking defaults if not in a real config
            CHUNK_LENGTH_MIN = 30
            CHUNK_OVERLAP_SEC = 5
            TEMP_CHUNK_DIR = "temp_audio_chunks_phase0"

        config = PlaceholderConfig()
        print("Warning: Using placeholder config due to import error in phase0_transcribe_audio.py.")

logger = logging.getLogger(__name__)

# --- Configuration (specific to this script, but using config for main paths) ---
# These could also be moved to config.py if preferred for universal configuration
TRANSCRIPTION_PROMPT = "Transcribe this audio recording. Provide only the text of the speech."


def setup_logging(day_identifier: str, verbose: bool = False):
    """Sets up logging for this script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_directory = config.LOG_DIR # Use the central log directory from config
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Include day_identifier in the log filename for uniqueness and clarity
    logfile_name = f"phase0_transcribe_audio_{day_identifier}_{ts}.log"
    logfile = log_directory / logfile_name

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.info("Logging initialised for Phase 0 (Audio Transcription) -> %s", logfile)


def get_audio_duration_ffmpeg(file_path: Path) -> float | None:
    """Gets the duration of an audio file in seconds using ffprobe."""
    command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(file_path)]
    try:
        logger.info(f"Running ffprobe to get duration for: {file_path.name}")
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        metadata = json.loads(result.stdout)
        if 'format' in metadata and 'duration' in metadata['format']:
            return float(metadata['format']['duration'])
        if 'streams' in metadata and metadata['streams']:
            for stream in metadata['streams']:
                if 'duration' in stream and stream.get('codec_type') == 'audio':
                    return float(stream['duration'])
        logger.warning(f"Could not find duration in ffprobe output for {file_path.name}")
        return None
    except FileNotFoundError:
        logger.error("Error: ffprobe command not found. Ensure FFmpeg (with ffprobe) is installed and in PATH.")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe for {file_path.name}: {e}")
        logger.error(f"ffprobe stderr: {e.stderr}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error parsing ffprobe output for {file_path.name}.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting audio duration for {file_path.name}: {e}")
        return None

def _configure_gemini_for_transcription():
    """Configures Gemini SDK for transcription, using universal model."""
    if not genai: return None # SDK not imported
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables. Cannot configure Gemini.")
        return None
    try:
        genai.configure(api_key=api_key)
        model_name = getattr(config, 'UNIVERSAL_GEMINI_MODEL_NAME', "gemini-1.5-pro-preview-0409") # Fallback
        logger.info(f"Configuring Gemini for transcription with model: {model_name}")
        return genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Error configuring Gemini SDK for transcription: {e}")
        return None

def _upload_audio_chunk_to_gemini(file_path: Path):
    """Uploads an audio chunk to Gemini and returns the file object."""
    if not genai: return None
    logger.info(f"Uploading {file_path.name} to Gemini... This may take a moment.")
    try:
        audio_file = genai.upload_file(path=file_path)
        while audio_file.state.name == "PROCESSING":
            logger.info(f"File {file_path.name} is processing on Gemini...")
            time.sleep(10)
            audio_file = genai.get_file(name=audio_file.name)
            if audio_file.state.name == "FAILED":
                logger.error(f"File processing failed for {file_path.name} on Gemini. Reason: {audio_file.state}")
                return None
        if audio_file.state.name == "ACTIVE":
            logger.info(f"File {file_path.name} uploaded and active successfully on Gemini.")
            return audio_file
        else:
            logger.error(f"File {file_path.name} not active after processing. State: {audio_file.state.name}")
            return None
    except Exception as e:
        logger.error(f"Error uploading file {file_path.name} to Gemini: {e}")
        return None

def _transcribe_uploaded_chunk(model, uploaded_audio_file) -> str:
    """Sends an uploaded audio file to Gemini for transcription."""
    if not model or not uploaded_audio_file:
        return "[Error: Model or uploaded file not available for transcription]"
    logger.info(f"Requesting transcription for {uploaded_audio_file.display_name}...")
    try:
        response = model.generate_content([TRANSCRIPTION_PROMPT, uploaded_audio_file], stream=False)
        response.resolve()
        text_parts = [part.text for part in response.parts if hasattr(part, 'text')]
        transcript = "".join(text_parts).strip()
        if not transcript:
            logger.warning(f"No text found in transcription response for {uploaded_audio_file.display_name}.")
            return f"[No text transcribed from {uploaded_audio_file.display_name}]"
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription for {uploaded_audio_file.display_name}: {e}")
        return f"[Error transcribing {uploaded_audio_file.display_name}: {e}]"

def main() -> int:
    """
    Main function for Phase 0: Audio Transcription.
    Processes a single audio file specified in config.DAY1_RAW_AUDIO_FILE.
    Outputs transcript to config.MASTER_TRANSCRIPT_FILE.
    """

    # --- DEBUG LINES START ---
    import sys # Ensure sys is available
    print("--- DEBUG: phase0_transcribe_audio.py main() function entered ---", file=sys.stderr)
    sys.stderr.flush()
    # --- DEBUG LINES END ---
    parser = argparse.ArgumentParser(description="Phase 0: Transcribe audio for a specific day.")
    parser.add_argument("--day", type=str, required=True, help="Day identifier (e.g., 'day_1', 'day_2')")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for this script."
    )
    args = parser.parse_args()
    day_identifier = args.day

    # Re-initialize logging with verbosity and day_identifier. This MUST be after args are parsed.
    setup_logging(day_identifier=args.day, verbose=args.verbose)

    logger.info(f"=== Phase 0: Audio Transcription started for {day_identifier} ===")
    if args.verbose:
        logger.debug(f"Verbose logging enabled for Phase 0 script (day: {day_identifier}).")
    # Use day-specific paths from config
    input_audio_paths = config.get_raw_audio_files(day_identifier) # Will be a list
    output_transcript_path = config.get_master_transcript_file(day_identifier)

    current_processing_audio_path: Path | None = None
    # These will be inside temp_processing_dir, so cleanup of that dir handles them.
    # temp_concatenated_audio_path: Path | None = None 
    # concat_list_file_path: Path | None = None

    # Define the main temporary directory for this phase early.
    # This will hold concatenated files (if any) and audio chunks.
    temp_processing_dir = config.get_temp_audio_chunks_dir(day_identifier)

    # Idempotency check: Skip if output already exists and is non-empty
    if output_transcript_path.exists() and output_transcript_path.stat().st_size > 0:
        logger.info(f"Master transcript {output_transcript_path} for {day_identifier} already exists and is not empty. Skipping Phase 0.")
        return 0 # Indicate success as no action needed

    if not genai:
        logger.critical("Google Generative AI SDK (google-generativeai) is not installed. Cannot proceed.")
        return 1

    if load_dotenv:
        env_path = config.PROJECT_ROOT / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env file from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}. GOOGLE_API_KEY might not be loaded if not set globally.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables. Aborting Phase 0.")
        return 1

    # --- Prepare temp directory for all audio operations (concatenation, chunks) ---
    if temp_processing_dir.exists():
        try:
            shutil.rmtree(temp_processing_dir)
            logger.info(f"Cleaned up existing temporary processing directory: {temp_processing_dir}")
        except Exception as e_shutil:
             logger.warning(f"Could not remove existing temp processing dir {temp_processing_dir}: {e_shutil}")
    try:
        temp_processing_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temporary processing directory: {temp_processing_dir}")
    except Exception as e_mkdir:
        logger.error(f"Could not create temp processing dir {temp_processing_dir}: {e_mkdir}. Aborting.")
        return 1

    # --- Handle input audio files: use single file or concatenate multiple parts ---
    if not input_audio_paths:
        logger.error(f"Error: No input audio files found for day '{day_identifier}' using pattern '{day_identifier}_recording.*' in {config.MATERIALS_DIR}.")
        return 1
    
    if len(input_audio_paths) == 1:
        current_processing_audio_path = input_audio_paths[0]
        logger.info(f"Using single input audio file for {day_identifier}: {current_processing_audio_path}")
    else: # Multiple files, need concatenation
        logger.info(f"Found {len(input_audio_paths)} audio parts for {day_identifier}. Concatenating them into {temp_processing_dir}.")
        
        first_file_suffix = input_audio_paths[0].suffix
        temp_concatenated_audio_path = temp_processing_dir / f"{day_identifier}_recording_concatenated{first_file_suffix}"
        concat_list_file_path = temp_processing_dir / f"{day_identifier}_concat_list.txt"

        try:
            with open(concat_list_file_path, 'w', encoding='utf-8') as f_list:
                for audio_file_part in input_audio_paths:
                    f_list.write(f"file '{audio_file_part.resolve()}'\n")
            logger.debug(f"Created concatenation list file: {concat_list_file_path}")

            concat_command = [
                "ffmpeg", "-y", 
                "-f", "concat",
                "-safe", "0", 
                "-i", str(concat_list_file_path),
                "-c", "copy", 
                str(temp_concatenated_audio_path)
            ]
            logger.info(f"Running FFmpeg for concatenation (codec copy): {' '.join(concat_command)}")
            result = subprocess.run(concat_command, capture_output=True, text=True, check=False, encoding='utf-8')

            if result.returncode != 0:
                logger.warning(f"FFmpeg concatenation with codec copy failed. Stderr: {result.stderr}. Stdout: {result.stdout}. Attempting re-encode.")
                concat_command_recode = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_list_file_path),
                    # No "-c copy", let ffmpeg choose appropriate audio codec for output container
                    str(temp_concatenated_audio_path)
                ]
                logger.info(f"Running FFmpeg for concatenation (re-encode): {' '.join(concat_command_recode)}")
                # For re-encoding, check=True is more appropriate as a last resort
                subprocess.run(concat_command_recode, capture_output=True, text=True, check=True, encoding='utf-8')
                logger.info(f"Successfully concatenated audio parts with re-encoding to: {temp_concatenated_audio_path}")
            else:
                logger.info(f"Successfully concatenated audio parts (codec copy) to: {temp_concatenated_audio_path}")
            
            current_processing_audio_path = temp_concatenated_audio_path

        except FileNotFoundError:
            logger.critical("ffmpeg command not found during concatenation. Please ensure FFmpeg is installed and in PATH. Aborting.")
            return 1
        except subprocess.CalledProcessError as e_ffmpeg_concat:
            logger.error(f"Error concatenating audio files with FFmpeg. Command: '{' '.join(e_ffmpeg_concat.cmd)}'. Stderr: {e_ffmpeg_concat.stderr}")
            return 1
        except Exception as e_concat:
            logger.error(f"Unexpected error during audio concatenation: {e_concat}")
            return 1
            
    if not current_processing_audio_path:
        logger.error("Internal error: current_processing_audio_path was not set after audio handling. Aborting.")
        return 1

    logger.info(f"Effective input audio for transcription for {day_identifier}: {current_processing_audio_path}")
    logger.info(f"Output transcript file for {day_identifier}: {output_transcript_path}")

    if not current_processing_audio_path.is_file():
        logger.error(f"Error: Effective input audio file '{current_processing_audio_path}' not found or is not a file.")
        return 1

    gemini_model = _configure_gemini_for_transcription()
    if not gemini_model:
        logger.error("Failed to configure Gemini model. Aborting Phase 0.")
        return 1

    total_duration_seconds = get_audio_duration_ffmpeg(current_processing_audio_path)
    if total_duration_seconds is None:
        logger.error(f"Could not determine duration of {current_processing_audio_path.name} for {day_identifier}. Cannot proceed with chunking.")
        return 1

    # temp_processing_dir is already prepared (cleaned and created) above.
    # It will be used as the directory for audio chunks.
    # No need to re-initialize or re-create temp_chunk_path_dir here.

    chunk_length_min = getattr(config, 'CHUNK_LENGTH_MIN', 30)
    overlap_sec = getattr(config, 'CHUNK_OVERLAP_SEC', 5)
    chunk_length_s = chunk_length_min * 60
    overlap_s = overlap_sec

    effective_chunk_processing_length = chunk_length_s - overlap_s
    if effective_chunk_processing_length <= 0:
        logger.error(f"Chunk length ({chunk_length_s}s) must be greater than overlap ({overlap_s}s). Aborting.")
        return 1

    num_chunks = 0
    if total_duration_seconds > 0:
        num_chunks = 1
        remaining_duration = total_duration_seconds - chunk_length_s
        if remaining_duration > 0:
            num_chunks += (remaining_duration + effective_chunk_processing_length - 1) // effective_chunk_processing_length
    if total_duration_seconds <= chunk_length_s: num_chunks = 1

    num_chunks_int = int(num_chunks)
    logger.info(f"Audio duration: {total_duration_seconds / 60:.2f} minutes. Number of chunks: {num_chunks_int}")

    if num_chunks_int == 0 and total_duration_seconds > 0: num_chunks_int = 1
    if num_chunks_int == 0 and total_duration_seconds == 0:
        logger.warning(f"Audio file {current_processing_audio_path.name} for {day_identifier} has zero duration. Writing empty transcript.")
        output_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        output_transcript_path.write_text(f"[Audio file {current_processing_audio_path.name} for {day_identifier} has zero duration]\\n", encoding='utf-8')
        return 0


    all_chunk_transcripts = []
    for i in range(num_chunks_int):
        start_s = i * effective_chunk_processing_length
        duration_to_extract_s = chunk_length_s
        if i == 0 and total_duration_seconds < overlap_s and total_duration_seconds > 0: start_s = 0
        if start_s + duration_to_extract_s > total_duration_seconds:
            duration_to_extract_s = total_duration_seconds - start_s
        if start_s >= total_duration_seconds: break
        if duration_to_extract_s <= 0: continue

        original_extension = current_processing_audio_path.suffix
        chunk_file_name = f"chunk_{day_identifier}_{i+1}{original_extension}" # Make chunk filename day-specific
        temp_chunk_file_path = temp_processing_dir / chunk_file_name # Use temp_processing_dir

        logger.info(f"Processing chunk {i+1}/{num_chunks_int} for {day_identifier} (Extracting from {start_s:.2f}s for {duration_to_extract_s:.2f}s)")
        ffmpeg_command = [
            "ffmpeg", "-y", "-i", str(current_processing_audio_path), # Use current_processing_audio_path
            "-ss", str(start_s), "-t", str(duration_to_extract_s),
            "-c", "copy", str(temp_chunk_file_path)
        ]
        try:
            logger.debug(f"Running FFmpeg: {' '.join(ffmpeg_command)}")
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=False, encoding='utf-8')
            if result.returncode != 0:
                logger.warning(f"FFmpeg (codec copy) failed for chunk {i+1}. Stderr: {result.stderr}. Retrying with re-encode.")
                ffmpeg_command_recode = [
                    "ffmpeg", "-y", "-i", str(current_processing_audio_path), # Use current_processing_audio_path
                    "-ss", str(start_s), "-t", str(duration_to_extract_s),
                    str(temp_chunk_file_path)
                ]
                logger.debug(f"Running FFmpeg (re-encode): {' '.join(ffmpeg_command_recode)}")
                subprocess.run(ffmpeg_command_recode, capture_output=True, text=True, check=True, encoding='utf-8')
            logger.info(f"Exported chunk to {temp_chunk_file_path}")
        except FileNotFoundError:
            logger.critical("ffmpeg command not found. Please ensure FFmpeg is installed and in PATH. Aborting.")
            return 1
        except subprocess.CalledProcessError as e_ffmpeg:
            logger.error(f"Error exporting chunk {i+1} with FFmpeg (re-encode): {e_ffmpeg.stderr}")
            all_chunk_transcripts.append(f"[Error exporting chunk {i+1}: {e_ffmpeg.stderr}]")
            continue
        except Exception as e_export:
            logger.error(f"Unexpected error exporting chunk {i+1}: {e_export}")
            all_chunk_transcripts.append(f"[Unexpected error exporting chunk {i+1}: {e_export}]")
            continue

        uploaded_chunk = _upload_audio_chunk_to_gemini(temp_chunk_file_path)
        if uploaded_chunk:
            transcript_text = _transcribe_uploaded_chunk(gemini_model, uploaded_chunk)
            all_chunk_transcripts.append(transcript_text)
            try: # Cleanup Gemini file
                genai.delete_file(uploaded_chunk.name)
                logger.info(f"Deleted uploaded chunk {uploaded_chunk.name} from Gemini.")
            except Exception as e_del_gemini:
                logger.warning(f"Could not delete chunk {uploaded_chunk.name} from Gemini: {e_del_gemini}")
        else:
            all_chunk_transcripts.append(f"[Upload or transcription failed for chunk {i+1}]")

        try: # Cleanup local chunk file
            os.remove(temp_chunk_file_path)
        except OSError as e_remove_local:
            logger.warning(f"Error deleting local chunk file {temp_chunk_file_path}: {e_remove_local}")

    final_transcript = "\n\n".join(all_chunk_transcripts)
    try:
        output_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_transcript_path, "w", encoding="utf-8") as f:
            f.write(final_transcript)
        logger.info(f"Successfully compiled and saved transcript to: {output_transcript_path}")
    except IOError as e_io:
        logger.error(f"Error writing final transcript to {output_transcript_path}: {e_io}")
        return 1
    finally: # Final cleanup of temp directory
        # temp_processing_dir now holds all temporary files (concatenated audio, list file, chunks)
        if temp_processing_dir.exists():
            try:
                shutil.rmtree(temp_processing_dir)
                logger.info(f"Cleaned up temporary processing directory: {temp_processing_dir}")
            except Exception as e_shutil_final:
                logger.warning(f"Error cleaning up final temp processing dir {temp_processing_dir}: {e_shutil_final}")


    if not output_transcript_path.exists() or output_transcript_path.stat().st_size == 0:
        logger.error(f"Phase 0 for {day_identifier} seems to have completed, but output transcript {output_transcript_path} is missing or empty.")
        return 1

    logger.info(f"=== Phase 0: Audio Transcription for {day_identifier} completed successfully ===")
    return 0

if __name__ == "__main__":
    # Ensure this script can be run as a module and directly for testing
    # The main orchestrator (run_pipeline.py) will call it as a module.
    # `python -m scripts.phase0_transcribe_audio`
    exit_code = main()
    sys.exit(exit_code)
