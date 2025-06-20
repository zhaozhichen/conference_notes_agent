import logging
from datetime import datetime
from pathlib import Path
import json
import os
import sys # Import sys for sys.stdout
# import subprocess # No longer needed if Pandoc is replaced
import argparse # For command-line arguments
import re
import shutil

# Attempt to import necessary modules
try:
    from dotenv import load_dotenv
except ImportError:
    logging.warning("python-dotenv not installed. .env file may not be loaded automatically by this script.")
    load_dotenv = None

try:
    # For web search for TODO items
    # Using googlesearch-python as a simple option, may need 'beautifulsoup4' and 'requests'
    # This library can be fragile due to scraping. A proper API might be better for production.
    from googlesearch import search as google_search
except ImportError:
    logging.warning("googlesearch-python not installed. Web search for TODO items will be skipped.")
    google_search = None

try:
    from weasyprint import HTML, CSS
    from weasyprint.logger import LOGGER as WEASYPRINT_LOGGER
    # Set WeasyPrint logger to a higher level to avoid verbose fontconfig messages, etc.
    WEASYPRINT_LOGGER.setLevel(logging.WARNING)
except ImportError:
    logging.error("WeasyPrint not installed. PDF generation will fail. Please install it (pip install weasyprint) and its system dependencies (pango, cairo, gdk-pixbuf).")
    HTML = None # To allow script to load but fail gracefully later
    CSS = None

try:
    import markdown2
except ImportError:
    logging.error("markdown2 not installed. Markdown to HTML conversion will fail. Please install it (pip install markdown2).")
    markdown2 = None

try:
    from scripts.utils import config
except ImportError:
    logging.error("CRITICAL: Could not import config.py. Ensure it's accessible and project is run as a module.")
    # Define critical fallbacks or exit
    class PlaceholderConfig:
        TODO_RAW_JSON_FILE = Path("../working_data/todo_raw.json")
        ENRICHED_NOTES_MD_FILE = Path("../working_data/enriched_notes.md")
        CROPPED_SLIDES_DIR = Path("../working_data/cropped_slides") # Used for path context
        FINAL_PDF_FILE = Path("../output/Conference_Notes_CCS25_Day1.pdf")
        SCRATCHPAD_DIR = Path("../working_data/scratchpad")
        PROJECT_ROOT = Path(__file__).resolve().parent.parent # Assumes this script is in scripts/
    config = PlaceholderConfig()
    logging.warning("Using placeholder config due to import error.")

logger = logging.getLogger(__name__)

def setup_logging(day_identifier: str, verbose: bool = False):
    """Sets up logging for this script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_directory = config.LOG_DIR # Use the central log directory

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_name = f"phase4_compilation_{day_identifier}_{ts}.log"
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
    logger.info(f"Logging initialised for Phase 4 (Compilation & Delivery) -> {logfile}")

def load_json_file(file_path: Path, description: str) -> list | dict | None:
    """Loads content from a JSON file, logs errors."""
    if not file_path.is_file():
        logger.error(f"{description} file not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            logger.info(f"Successfully loaded {description} from {file_path}")
            return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {description} file {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {description} file {file_path}: {e}")
        return None

def load_text_file(file_path: Path, description: str) -> str | None:
    """Loads content from a text file, logs errors."""
    if not file_path.is_file():
        logger.error(f"{description} file not found at: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            logger.info(f"Successfully loaded {description} from {file_path} (length: {len(content)} chars)")
            return content
    except Exception as e:
        logger.error(f"Error reading {description} file {file_path}: {e}")
        return None

def format_action_items(todo_items: list) -> str:
    """
    Formats the list of to-do items into a Markdown string for cleaner output.
    """
    if not todo_items:
        return "No action items identified.\n" # Single newline

    # Start with the header, followed by one blank line
    md_string = "## Action Items & Further Reading\n\n"
    
    item_lines = []
    for item_text in todo_items:
        if item_text and isinstance(item_text, str):
            item_lines.append(f"- [ ] {item_text}")
        else:
            logger.warning(f"Skipping invalid to-do item: {item_text}")
    
    md_string += "\n".join(item_lines) # Join items, each on a new line
    
    # Ensure there's content before adding trailing newlines if item_lines was not empty
    if item_lines:
        md_string += "\n\n" # Ensure one blank line after the list section
    else:
        # If there were no valid items, md_string is just the header. 
        # We might not want extra newlines if the list was empty or all items invalid.
        # Or, ensure at least one newline after header if list is empty.
        md_string += "\n" # Add a single newline if list part is empty.

    return md_string

def convert_obsidian_image_links_to_markdown(markdown_text: str, day_identifier: str) -> str:
    """Converts Obsidian-style image wikilinks ![[filename.jpg]] to standard Markdown ![]()."""
    logger.info("[convert_obsidian_links] Starting conversion of Obsidian image links.")
    cropped_slides_dir_for_day = config.get_cropped_slides_dir(day_identifier)

    def replace_link(match):
        # This is the body of the nested replace_link function.
        # Ensure this entire block is indented one level relative to "def replace_link(match):"
        image_filename_or_partial_path = match.group(1).strip()

        # Handle cases where the LLM might already include "cropped_slides/"
        if image_filename_or_partial_path.startswith("cropped_slides/"):
            image_filename_actual = image_filename_or_partial_path[len("cropped_slides/"):]
            logger.debug(f"[convert_obsidian_links] Handled prefixed path: '{image_filename_or_partial_path}' -> '{image_filename_actual}'")
        else:
            image_filename_actual = image_filename_or_partial_path

        # Construct absolute path to the image file
        absolute_image_path = cropped_slides_dir_for_day / image_filename_actual

        if absolute_image_path.is_file():
            # Convert to file:/// URL
            image_url = absolute_image_path.as_uri()
            logger.debug(f"[convert_obsidian_links] Converted '{match.group(0)}' to '![]({image_url})'")
            return f"![]({image_url})"
        else:
            logger.warning(f"[convert_obsidian_links] Image not found at expected absolute path: {absolute_image_path} (original link: {match.group(0)}). Using original link text as placeholder.")
            return f"![Image not found: {image_filename_or_partial_path}]({image_filename_or_partial_path})"
    # End of replace_link function body

    # THIS IS THE CORRECTED REGEX PATTERN:
    regex_pattern = r"!\[\[([^|\]]+?)(?:\|([^\]]*?))?\]\]"

    # --- DEBUG: Print the markdown_text and regex pattern ---
    logger.debug(f"Attempting to use regex_pattern: '{regex_pattern}'") # Print the regex pattern
    logger.debug(f"[convert_obsidian_links] Markdown text BEFORE re.sub (first 500 chars):\n{markdown_text[:500]}")
    if len(markdown_text) > 500:
        logger.debug("[convert_obsidian_links] ... (markdown_text truncated for log) ...")
    # --- END DEBUG ---

    converted_text = re.sub(regex_pattern, replace_link, markdown_text)
    logger.info("[convert_obsidian_links] Finished conversion of Obsidian image links.")
    return converted_text

def generate_pdf_from_markdown(markdown_content: str, output_pdf_path: Path, day_identifier: str):
    """
    Generates a PDF from Markdown content using WeasyPrint.
    Images are referenced relative to the project's working_data directory.
    """
    if not HTML or not markdown2:
        logger.error("WeasyPrint or markdown2 library not available. Cannot generate PDF.")
        return

    logger.info(f"Attempting to generate PDF: {output_pdf_path}")

    try:
        # Convert Markdown to HTML
        # Add 'fenced-code-blocks' and 'tables' extras for common Markdown features.
        # Add 'break-on-newline' to make single newlines in Markdown cause line breaks in HTML.
        html_content = markdown2.markdown(markdown_content, extras=["fenced-code-blocks", "tables", "break-on-newline"])
        logger.info("Markdown content converted to HTML for PDF generation.")
        # logger.debug(f"Generated HTML (first 500 chars):\\n{html_content[:500]}")

        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Create WeasyPrint HTML object.
        # With absolute file:/// URLs for images, base_url is less critical for image path resolution,
        # but can still be useful for other relative links if any.
        # Setting it to the project root or working_data_dir is usually safe.
        html_doc = HTML(string=html_content, base_url=config.PROJECT_ROOT.as_uri()) # Or config.WORKING_DATA_DIR.as_uri()

        # Optional: Add some basic CSS for styling
        # For more complex styling, an external CSS file is better.
        # Example: Ensure images don't overflow page width
        css = CSS(string="""
            @page { margin: 1in; }
            body { font-family: sans-serif; line-height: 1.5; }
            h1, h2, h3, h4, h5, h6 { margin-top: 1.2em; margin-bottom: 0.5em; line-height: 1.2; }
            p { margin-top: 0; margin-bottom: 0.8em; }
            img { max-width: 100%; height: auto; display: block; margin-left: auto; margin-right: auto; margin-top: 0.5em; margin-bottom: 0.5em; }
            blockquote { border-left: 3px solid #ccc; padding-left: 1em; margin-left: 0; font-style: italic; color: #555; }
            table { border-collapse: collapse; width: 90%; margin-bottom: 1em; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            ul, ol { padding-left: 2em; }
        """)

        html_doc.write_pdf(output_pdf_path, stylesheets=[css])
        logger.info(f"Successfully generated PDF: {output_pdf_path}")

    except FileNotFoundError as e: # Should not happen if libraries are checked
        logger.error(f"A required file was not found during PDF generation (should be caught earlier): {e}")
    except Exception as e:
        logger.error(f"An error occurred during PDF generation with WeasyPrint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Optionally, save the HTML that failed for debugging to day-specific working data
        day_working_data_dir = config.get_day_working_data_dir(day_identifier)
        failed_html_path = day_working_data_dir / f"failed_pdf_generation_input_{day_identifier}.html"
        try:
            with open(failed_html_path, "w", encoding="utf-8") as f_html_err:
                f_html_err.write(html_content if 'html_content' in locals() else "# Markdown to HTML conversion failed or not reached.")
            logger.info(f"Problematic HTML (or placeholder) saved to {failed_html_path} for debugging.")
        except Exception as e_save_html:
            logger.error(f"Could not save debug HTML: {e_save_html}")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Final Compilation & Delivery for a specific day.")
    parser.add_argument("--day", type=str, required=True, help="Day identifier (e.g., 'day_1', 'day_2')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()
    day_identifier = args.day

    setup_logging(day_identifier=day_identifier, verbose=args.verbose)

    logger.info(f"=== Phase 4: Final Compilation & Delivery started for {day_identifier} ===")
    if args.verbose:
        logger.debug(f"Verbose logging enabled for Phase 4 script (day: {day_identifier}).")

    # Load .env variables if dotenv is available
    if load_dotenv:
        env_path = getattr(config, 'PROJECT_ROOT', Path(".")) / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env file from {env_path}")

    # 1. Load inputs from Phase 3
    todo_raw_json_file_path = config.get_todo_raw_json_file(day_identifier)
    enriched_notes_md_file_path = config.get_enriched_notes_md_file(day_identifier)
    final_pdf_file_path = config.get_final_pdf_file(day_identifier)

    todo_items_data = load_json_file(todo_raw_json_file_path, "To-Do items JSON")
    enriched_notes_md_str = load_text_file(enriched_notes_md_file_path, "Enriched Markdown notes")

    if todo_items_data is None: # Can be an empty list, but None means file load error
        logger.warning("To-Do items file could not be loaded. Proceeding without an Action Items section.")
        todo_items_data = [] # Default to empty list
    if not isinstance(todo_items_data, list):
        logger.warning(f"To-Do items data is not a list (type: {type(todo_items_data)}). Treating as empty.")
        todo_items_data = []

    if enriched_notes_md_str is None:
        logger.error(f"Enriched Markdown notes ({enriched_notes_md_file_path.name}) could not be loaded. Cannot generate final PDF for {day_identifier}.")
        enriched_notes_md_str = "# Error: Enriched notes content not available.\\n" # Fallback content

    # 2. Compile "Action Items" Section
    action_items_md_str = format_action_items(todo_items_data)

    # 3. Prepend Action Items to Enriched Notes and convert Obsidian image links
    logger.info("Converting Obsidian image links to standard Markdown...")
    notes_with_standard_links = convert_obsidian_image_links_to_markdown(enriched_notes_md_str, day_identifier)

    final_markdown_content = action_items_md_str + "\n" + notes_with_standard_links
    logger.info("Action items prepended to enriched notes.")
    logger.debug(f"First 500 chars of final Markdown for PDF:\\n{final_markdown_content[:500]}")

    # 4. Generate Final PDF Document
    generate_pdf_from_markdown(final_markdown_content, final_pdf_file_path, day_identifier)

    logger.info(f"=== Phase 4 for {day_identifier} completed ===")

if __name__ == "__main__":
    main()
