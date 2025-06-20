from pathlib import Path
import logging
from datetime import datetime
import os # For environment variables

# Attempt to import necessary modules, but allow failure if not fully configured yet
try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not installed. .env file will not be loaded by slide_extractor.")
    load_dotenv = None

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Gemini functionality will not be available.")
    genai = None

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    # pillow_heif is no longer needed as HEIC files are not used for timestamps
    import time # For file modification time fallback
except ImportError:
    print("Warning: Pillow not installed. Image processing and JPEG metadata extraction may fail.")
    # We don't need a pillow_heif fallback anymore

# It's good practice to import from .config to get paths and model names
try:
    from .config import (
        # RAW_SLIDES_DIR, # This will now be fetched by get_raw_slides_dir(day_identifier)
        get_raw_slides_dir,
        get_extracted_slide_content_file, # Import the function
        GEMINI_VISION_MODEL,
        # OCR_PROMPT, # Will be imported directly from .prompts
        PROJECT_ROOT, # Assuming PROJECT_ROOT is defined in config.py
        SUPPORTED_IMAGE_EXTS # For finding JPGs
    )
    from .prompts import OCR_PROMPT # Import OCR_PROMPT directly
except ImportError:
    print("Error: Could not import from .config in slide_extractor.py. Ensure config.py is correctly set up in the same directory (utils).")
    # Define fallbacks or raise an error if config is critical
    # For skeleton, we can define some placeholders to allow script to be imported
    # RAW_SLIDES_DIR = Path("../materials/day_1_slides") # Placeholder
    def get_raw_slides_dir(day_identifier: str): return Path(f"../materials/{day_identifier}_slides") # Placeholder function
    # Placeholder function for get_extracted_slide_content_file
    def get_extracted_slide_content_file(day_identifier: str):
        return Path(f"../working_data/{day_identifier}/extracted_slide_content.txt")
    GEMINI_VISION_MODEL = "gemini-pro-vision" # Placeholder
    # OCR_PROMPT placeholder is no longer needed here if direct import from prompts works
    # If prompts.py itself fails to import, this script will have bigger issues.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Placeholder
    SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg']


logger = logging.getLogger(__name__)

# --- JPEG Timestamp Extraction ---
def get_image_timestamp(image_path: Path) -> str:
    """
    Extracts the capture timestamp from a JPEG file's EXIF data.
    Falls back to file modification time if EXIF data is not found or usable.
    Format: YYYY-MM-DD HH:MM:SS
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif() # Returns a dictionary of EXIF tags

        if exif_data:
            # Common EXIF tags for date/time
            # 36867: DateTimeOriginal, 36868: DateTimeDigitized, 306: DateTime
            datetime_str = None
            for tag_id in (36867, 36868, 306):
                if tag_id in exif_data:
                    datetime_str = exif_data[tag_id]
                    # Sometimes it's a tuple of strings, sometimes a single string
                    if isinstance(datetime_str, tuple):
                        datetime_str = datetime_str[0]
                    break
            
            if datetime_str:
                # EXIF datetime format is typically 'YYYY:MM:DD HH:MM:SS'
                try:
                    dt_obj = datetime.strptime(str(datetime_str), '%Y:%m:%d %H:%M:%S')
                    logger.info(f"[slide_extractor] Found EXIF timestamp for {image_path.name}: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')}")
                    return dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    logger.warning(f"[slide_extractor] Could not parse EXIF datetime string '{datetime_str}' from {image_path.name}. Falling back to file modification time.")
        else:
            logger.warning(f"[slide_extractor] No EXIF data found in {image_path.name}. Falling back to file modification time.")

    except FileNotFoundError:
        logger.error(f"[slide_extractor] Image file not found for timestamp extraction: {image_path}")
        # Fall through to file modification time of the path if it exists, though unlikely if open failed.
        # More robustly, this might indicate a problem to return a specific error string.
    except Exception as e:
        logger.error(f"[slide_extractor] Error reading EXIF data for {image_path.name}: {e}. Falling back to file modification time.")

    # Fallback to file modification time
    try:
        mtime = image_path.stat().st_mtime
        dt_obj = datetime.fromtimestamp(mtime)
        ts_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"[slide_extractor] Using file modification time for {image_path.name}: {ts_str}")
        return ts_str + " (File ModTime)"
    except Exception as e_stat:
        logger.error(f"[slide_extractor] Could not get file modification time for {image_path.name}: {e_stat}")
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " (Error: Current Time)"


# --- Gemini OCR and Content Analysis (Placeholder/Basic) ---
def extract_content_from_slide_image_gemini(image_path: Path) -> dict:
    """
    Uses Gemini to extract text, table structures, and diagram descriptions.
    This is a placeholder and needs robust implementation.
    """
    if not genai: # Check if the genai module was imported successfully
        logger.error("[slide_extractor] Google Generative AI SDK (google-generativeai) not available. Cannot perform OCR with Gemini.")
        return {"text": "Error: Gemini SDK not available", "tables": [], "diagram_description": ""}

    try:
        # Load API Key - main script should ideally handle dotenv loading once
        if load_dotenv:
            # Assuming .env is at PROJECT_ROOT
            dotenv_path = PROJECT_ROOT / ".env"
            if dotenv_path.exists():
                load_dotenv(dotenv_path)
            else:
                logger.warning(f"[slide_extractor] .env file not found at {dotenv_path}. GOOGLE_API_KEY might not be loaded.")

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("[slide_extractor] GOOGLE_API_KEY not found in environment. Cannot authenticate Gemini.")
            return {"text": "Error: GOOGLE_API_KEY not configured", "tables": [], "diagram_description": ""}

        genai.configure(api_key=api_key)

        logger.info(f"[slide_extractor] Opening image for Gemini: {image_path.name}")
        img = Image.open(image_path)

        model = genai.GenerativeModel(GEMINI_VISION_MODEL)
        logger.info(f"[slide_extractor] Sending {image_path.name} to Gemini model: {GEMINI_VISION_MODEL} with prompt...")

        # For more complex structured output, you might need to use specific response schema handling
        # This basic prompt asks for text, tables (Markdown), and diagram description.
        # The model's ability to perfectly follow this depends on its capabilities.
        response = model.generate_content([OCR_PROMPT, img], stream=False)
        response.resolve() # Ensure all parts are available if not streaming

        extracted_text = ""
        # Try to access text. Response structure can vary.
        if hasattr(response, 'text'):
            extracted_text = response.text
        elif response.parts:
            extracted_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))

        # Placeholder for table and diagram extraction - this would require more sophisticated parsing of `extracted_text`
        # or more specific prompts to Gemini to return structured data (e.g., JSON for tables/diagrams).
        # For now, we assume the model might include Markdown tables or diagram descriptions in the main text.
        tables_md = [] # e.g., ["| Col A | Col B |\\n|---|---|\\n| Val 1 | Val 2 |"]
        diagram_desc = ""

        # Example: crude search for Markdown table in extracted_text
        if "|\n|---" in extracted_text: # Basic check for a markdown table
             # This is very naive; proper parsing or structured output from Gemini is better.
            tables_md.append("Table found (see full text for Markdown)")


        logger.info(f"[slide_extractor] Content extracted from {image_path.name} using Gemini.")
        return {
            "text": extracted_text.strip(),
            "tables": tables_md, # This would be a list of Markdown table strings
            "diagram_description": diagram_desc # This would be a text description
        }

    except FileNotFoundError:
        logger.error(f"[slide_extractor] Image file not found for Gemini processing: {image_path}")
        return {"text": f"Error: Image file not found - {image_path.name}", "tables": [], "diagram_description": ""}
    except Exception as e:
        logger.error(f"[slide_extractor] Error during Gemini content extraction for {image_path.name}: {e}")
        return {"text": f"Error during Gemini processing: {e}", "tables": [], "diagram_description": ""}

# --- Main Processing Function ---
def process_slides(day_identifier: str) -> list:
    """
    Main function for Phase 1: Slide Content Extraction.
    - Iterates through raw slide images (.jpg) for a specific day.
    - For each, extracts content using Gemini.
    - Retrieves timestamp from the JPEG file itself.
    - Writes formatted content to the day-specific extracted slide content file.
    - Returns a list of records (dictionaries) of the extracted content.

    Args:
        day_identifier (str): The identifier for the day (e.g., "day_1", "day_2").
    """
    logger.info(f"[slide_extractor] Starting slide content extraction process for {day_identifier}...")

    current_raw_slides_dir = get_raw_slides_dir(day_identifier)
    logger.info(f"[slide_extractor] Using raw slides directory: {current_raw_slides_dir}")

    if not current_raw_slides_dir.exists():
        logger.error(f"[slide_extractor] Raw slides directory not found: {current_raw_slides_dir} for {day_identifier}. Aborting.")
        return []
    current_extracted_content_file = get_extracted_slide_content_file(day_identifier)

    # Ensure output directory for EXTRACTED_SLIDE_CONTENT_FILE exists
    current_extracted_content_file.parent.mkdir(parents=True, exist_ok=True)

    extracted_records = []
    processed_files_count = 0

    # Iterate through JPG files for OCR from the day-specific directory
    jpg_files = [f for f in current_raw_slides_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]

    if not jpg_files:
        logger.warning(f"[slide_extractor] No supported image files (e.g., .jpg) found in {current_raw_slides_dir} for {day_identifier}.")
        # Create an empty output file if no images are found
        with open(current_extracted_content_file, "w", encoding="utf-8") as f:
            f.write("# No images processed or no content extracted.\\n")
        logger.info(f"[slide_extractor] Created empty/placeholder output file: {current_extracted_content_file}")
        return []

    with open(current_extracted_content_file, "w", encoding="utf-8") as outfile:
        logger.info(f"[slide_extractor] Writing extracted content to: {current_extracted_content_file}")

        for jpg_path in sorted(jpg_files):
            logger.info(f"[slide_extractor] Processing slide: {jpg_path.name}")

            # 1. Extract content using Gemini (OCR, tables, diagrams)
            content_data = extract_content_from_slide_image_gemini(jpg_path)

            # 2. Get timestamp from the JPEG file itself
            timestamp_str = get_image_timestamp(jpg_path)
            # get_image_timestamp has its own fallbacks, so timestamp_str should always be a string

            # 3. Format and write to output file
            outfile.write(f"[{timestamp_str}] Slide: {jpg_path.name}\n")
            outfile.write("TEXT:\n")
            outfile.write(content_data.get("text", "N/A") + "\n\n")

            if content_data.get("tables"):
                outfile.write("TABLES:\n")
                for table_md in content_data["tables"]:
                    outfile.write(table_md + "\n\n") # Assuming tables are Markdown strings

            if content_data.get("diagram_description"):
                outfile.write("DIAGRAM DESCRIPTION:\n")
                outfile.write(content_data["diagram_description"] + "\n\n")

            outfile.write("---\n\n") # Separator

            # 4. Store record (optional, if needed by calling script)
            record = {
                "filename": jpg_path.name,
                "timestamp": timestamp_str,
                "text": content_data.get("text"),
                "tables": content_data.get("tables"),
                "diagram_description": content_data.get("diagram_description")
            }
            extracted_records.append(record)
            processed_files_count += 1
            logger.info(f"[slide_extractor] Finished processing {jpg_path.name}")

    if processed_files_count > 0:
        logger.info(f"[slide_extractor] Successfully processed {processed_files_count} slide images for {day_identifier}.")
    else:
        logger.warning(f"[slide_extractor] No slide images were successfully processed with content extraction for {day_identifier}.")
        # Ensure file exists even if loop didn't run or failed early
        if not current_extracted_content_file.exists():
             with open(current_extracted_content_file, "w", encoding="utf-8") as f:
                f.write("# No images processed or no content extracted.\\n")

    return extracted_records

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    # This requires config.py to be correctly set up and .env for API key

    # Setup basic logging for standalone test
    # The main script (1_preprocess_slides.py) should set up logging for the whole phase.
    # This is just for direct testing of slide_extractor.
    test_log_dir = PROJECT_ROOT / "working_data" / "scratchpad" # from .config idea
    test_log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[
            logging.FileHandler(test_log_dir / "slide_extractor_test.log"),
            logging.StreamHandler()
        ]
    )

    logger.info("Running slide_extractor.py example...")

    # Ensure materials/day_1_slides directory and a sample JPG/HEIC exist for testing.
    # Example: Create dummy files if they don't exist.
    # This is complex to do robustly here without more context on actual files.
    # For a real test, you'd have sample files in your test setup.

    # RAW_SLIDES_DIR and other configs should be loaded from .config
    # Ensure config.py is importable and correct.
    test_day_id = "day_1" # Example for testing
    test_raw_slides_dir = get_raw_slides_dir(test_day_id)
    print(f"Using RAW_SLIDES_DIR for {test_day_id}: {test_raw_slides_dir}")
    test_output_file = get_extracted_slide_content_file(test_day_id)
    print(f"Outputting to day-specific extracted content file: {test_output_file}")

    if not test_raw_slides_dir.exists():
        print(f"WARNING: Test mode - RAW_SLIDES_DIR {test_raw_slides_dir} for {test_day_id} does not exist. Creating for test.")
        test_raw_slides_dir.mkdir(parents=True, exist_ok=True)
        # Consider adding a dummy JPG here for a more complete test.
        # For now, it will likely process 0 files if the directory is empty.

    results = process_slides(day_identifier=test_day_id) # Pass the day_identifier for testing
    if results:
        print(f"\nTest successful for {test_day_id}. Extracted records for {len(results)} slides.")
        # print("First record:", results[0])
    else:
        print("\nTest run completed. No records extracted, or an error occurred. Check logs.")
