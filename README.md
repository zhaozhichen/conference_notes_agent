# AI Conference Note-Taking Assistant

## Project Overview

This project provides an automated AI-powered pipeline to process, contextualize, and synthesize raw conference materials (audio recordings, presentation slides, and user's manual notes) into a single, enriched, and actionable set of notes for a specified day. The goal is to create a polished PDF document that integrates the user's observations with detailed information extracted and processed by Large Language Models (LLMs).

The system is designed to handle day-specific data, making it easy to manage notes for multi-day events. It utilizes centralized logging and a modular, multi-phase architecture.

## Features

*   **Automated Audio Transcription:** (Optional) Transcribes raw audio recordings using AI.
*   **Slide Content Extraction:** Extracts text and describes visual elements (tables, diagrams) from slide images using an LLM.
*   **Image Processing:** Standardizes slide images by cropping and resizing them.
*   **AI-Powered Contextualization:** Maps slides to conference sessions and segments audio transcripts based on the agenda and user notes.
*   **Note Synthesis & Enrichment:** Uses an LLM to enrich the user's manual notes by integrating relevant details from slides and audio, resolving placeholders like `[slides]`, `[audio]`, and `[todo]`.
*   **Action Item Extraction:** Identifies and compiles a list of to-do items from the notes.
*   **PDF Generation:** Compiles the enriched notes and action items into a final, professionally formatted PDF document.
*   **Day-Specific Processing:** Organizes inputs, intermediate files, and outputs based on the conference day.
*   **Centralized Logging:** Provides detailed logs for each phase of the pipeline.

## Directory Structure

```
confidential_computing_summit/
├── .env                  # Environment variables (GOOGLE_API_KEY)
├── materials/            # Input raw materials (agenda, notes, audio recordings, slides)
│   ├── agenda.txt
│   ├── day_1_notes.txt
│   ├── day_1_recordings/ # Preferred location for audio files for a day
│   │   └── day_1_recording_part_01.m4a
│   ├── day_1_recording_transcript.txt # Transcript of the day's audio
│   └── day_1_slides/
│       └── IMG_1234.jpeg
├── scripts/              # Python scripts for each pipeline phase
│   ├── phase0_transcribe_audio.py
│   ├── phase1_asset_processing.py
│   ├── phase2_contextualization.py
│   ├── phase3_synthesis.py
│   ├── phase4_compilation.py
│   └── utils/            # Utility modules (config, prompts, etc.)
├── working_data/         # Intermediate files generated during processing (organized by day)
│   └── day_1/
│       ├── extracted_slide_content.txt
│       ├── cropped_slides/
│       └── ...
├── output/               # Final output PDF documents (organized by day)
│   └── Conference_Notes_CCS25_day_1.pdf
├── logs/                 # Log files for each script execution
├── run_pipeline.py       # Main script to orchestrate the pipeline
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── ...                   # Other project files
```

## Setup

### Prerequisites

*   Python (version 3.8 or higher recommended)
*   `pip` (Python package installer)
*   `ffmpeg`: Required for audio processing in Phase 0. Ensure it's installed and accessible in your system's PATH.
    *   On macOS (using Homebrew): `brew install ffmpeg`
    *   On Linux (Debian/Ubuntu): `sudo apt update && sudo apt install ffmpeg`
    *   On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd confidential_computing_summit
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Environment Variables

The pipeline requires an API key for Google's Generative AI services (e.g., Gemini).

1.  Create a file named `.env` in the root of the project directory (`confidential_computing_summit/`).
2.  Add your Google API key to this file:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"
    ```
    Replace `"YOUR_GOOGLE_AI_STUDIO_API_KEY"` with your actual API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Input Data Format

Place your raw conference materials in the `confidential_computing_summit/materials/` directory. The pipeline expects the following structure:

*   **`materials/agenda.txt`**: A text file containing the overall conference agenda/schedule. This is used to identify session titles and times.
*   **For each day of the conference (e.g., `day_1`, `day_2`):**
    *   **User's Manual Notes:** `materials/{day_identifier}_notes.txt` (e.g., `materials/day_1_notes.txt`)
        *   Text file containing your personal notes, observations, questions.
        *   Use placeholders:
            *   `[slides]`: To indicate where slide content should be inserted.
            *   `[audio]`: To indicate where audio transcript snippets are relevant.
            *   `[todo]`: To mark action items.
    *   **Audio Recording(s):**
        *   **Preferred location:** `materials/{day_identifier}_recordings/` (e.g., `materials/day_3_recordings/`)
            *   Create a subdirectory for each day's audio recordings.
            *   Place all audio files for that day (e.g., `part_01.m4a`, `session_A.mp3`, `day_3_main_talk.aac`) within this folder.
            *   All supported audio files (e.g., `.m4a`, `.aac`, `.mp3`, `.wav`) found in this directory will be sorted alphabetically by filename and then concatenated for transcription.
        *   **Fallback location:** `materials/{day_identifier}_recording*.<ext>` (e.g., `materials/day_1_recording.aac`, `materials/day_1_recording_part_01.m4a`).
            *   If the `{day_identifier}_recordings/` subdirectory is not found for a given day, the pipeline will attempt to find audio files directly under the `materials/` directory that match this pattern (e.g., `day_1_recording.aac`, `day_1_recording_part_01.m4a`, `day_1_recording_part_02.m4a`).
        *   Raw audio files from the sessions. Multiple parts are supported and will be concatenated.
        *   This is used by Phase 0 if you want the pipeline to perform transcription.
    *   **Audio Transcript (Optional):** `materials/{day_identifier}_recording_transcript.txt` (e.g., `materials/day_1_recording_transcript.txt`)
        *   If you have a pre-existing transcript, place it here. Phase 0 will be skipped if this file exists and is not empty.
        *   If not provided, Phase 0 will attempt to generate it from the raw audio recording(s).
    *   **Presentation Slides:** `materials/{day_identifier}_slides/` (e.g., `materials/day_1_slides/`)
        *   A directory containing images of the presentation slides (e.g., `.jpeg`, `.jpg`, `.png`).
        *   These images will be processed for content extraction and standardization.

## How to Run the Pipeline

Execute the main pipeline script `run_pipeline.py` from the root of the project directory (`confidential_computing_summit/`).

You need to specify the day you want to process using the `--day` argument.

```bash
python run_pipeline.py --day day_1
```

**Optional arguments:**

*   `--verbose`: Enables more detailed logging output to the console.
    ```bash
    python run_pipeline.py --day day_1 --verbose
    ```
*   `--force-phase <phase_number>`: Forces re-running a specific phase even if its outputs exist. For example, to force re-running Phase 1:
    ```bash
    python run_pipeline.py --day day_1 --force-phase 1
    ```
*   `--skip-phase <phase_number>`: Skips a specific phase. For example, to skip Phase 0:
    ```bash
    python run_pipeline.py --day day_1 --skip-phase 0
    ```

The final PDF output will be saved in the `output/` directory, named like `Conference_Notes_CCS25_{day_identifier}.pdf`.
Intermediate files are stored in `working_data/{day_identifier}/`, and logs are in the `logs/` directory.

## Pipeline Logic Flow

The pipeline is divided into several phases, each handled by a dedicated script:

1.  **Phase 0: Audio Transcription (`scripts.phase0_transcribe_audio`)** (Optional)
    *   **Input:** Raw audio file(s) for the specified day (e.g., `materials/day_1_recording.aac`).
    *   **Action:** Chunks audio, transcribes using an LLM, and consolidates transcripts.
    *   **Output:** Master transcript file (e.g., `materials/day_1_recording_transcript.txt`).
    *   *Skipped if a non-empty master transcript file already exists.*

2.  **Phase 1: Asset Pre-processing & Extraction (`scripts.phase1_asset_processing`)**
    *   **Slide Content Extraction:**
        *   **Input:** Slide images (e.g., `materials/day_1_slides/*.jpg`).
        *   **Action:** Extracts text, tables, and diagram descriptions from slides using an LLM (Gemini Vision). Timestamps are extracted from EXIF or file modification time.
        *   **Output:** Consolidated slide content text file (e.g., `working_data/day_1/extracted_slide_content.txt`).
    *   **Slide Image Preparation:**
        *   **Input:** Slide images.
        *   **Action:** Crops, resizes, and standardizes slide images.
        *   **Output:** Processed images (e.g., `working_data/day_1/cropped_slides/`).

3.  **Phase 2: Contextualization & Mapping (`scripts.phase2_contextualization`)**
    *   **Session Identification & Mapping:**
        *   **Input:** Agenda, user notes, extracted slide content.
        *   **Action (LLM-driven):** Identifies attended sessions, maps slides to sessions.
        *   **Output:** Slide-session mapping JSON (e.g., `working_data/day_1/slide_session_mapping.json`), mapped slide content text file.
    *   **Audio Transcript Segmentation:**
        *   **Input:** Master transcript, agenda, identified sessions.
        *   **Action (LLM-driven):** Segments the transcript by session.
        *   **Output:** Segmented audio transcript text (e.g., `working_data/day_1/llm_raw_audio_segmentation_output.txt`).

4.  **Phase 3: Synthesis & Enrichment (`scripts.phase3_synthesis`)**
    *   **Input:** User's manual notes, mapped slide content, segmented audio transcripts.
    *   **Action (LLM-driven):**
        *   Uses manual notes as a backbone.
        *   Resolves `[slides]` placeholders with slide content and image links.
        *   Resolves `[audio]` placeholders with audio snippets.
        *   Enriches notes with details from audio.
        *   Extracts `[todo]` items.
    *   **Output:** Enriched notes in Markdown (e.g., `working_data/day_1/enriched_notes.md`), to-do items JSON (e.g., `working_data/day_1/todo_raw.json`).

5.  **Phase 4: Final Compilation & Delivery (`scripts.phase4_compilation`)**
    *   **Compile Action Items:**
        *   **Input:** To-do items JSON.
        *   **Action:** Formats to-do items into a Markdown checklist.
    *   **Generate Final Document:**
        *   **Input:** Enriched Markdown notes, formatted action items.
        *   **Action:** Prepends action items, converts Obsidian image links to standard Markdown (`file:///` URLs), converts Markdown to PDF.
        *   **Output:** Final PDF document (e.g., `output/Conference_Notes_CCS25_day_1.pdf`).

## Tips on AI-Enhanced Note Taking

This pipeline is designed to augment your note-taking process, not replace your active engagement. Here's how to maximize its benefits:

*   **Focus Your Manual Notes:**
    *   **Human Strengths:** Use your manual notes (`{day}_notes.txt`) to capture high-level insights, personal reflections, "aha!" moments, connections to your existing knowledge, strategic questions, and key takeaways. These are things an AI might miss or can't infer.
    *   **Ideas & Thoughts:** Jot down novel ideas sparked by the presentation or discussion.
    *   **Actionable Items:** Clearly mark tasks with `[todo]`. Be specific (e.g., `[todo] Read paper by Dr. Smith on X`, `[todo] Email contact@example.com about Y`).
    *   **Q&A:** Note down interesting questions asked (by you or others) and the answers provided.

*   **Let the AI Handle the Details:**
    *   **Slides (`[slides]`):** Don't frantically copy slide content. Simply put `[slides]` where you want the relevant slide information to appear. The AI will extract text, describe diagrams, and even link the image. You can add a brief note like `[slides] - important diagram on architecture` to give context.
    *   **Audio (`[audio]`):** If the speaker says something profound or you need a verbatim quote, mark it with `[audio]`. The AI will try to find relevant segments from the transcript. You can add keywords like `[audio] - speaker's definition of confidential attestations`.
    *   **Broad Enrichment:** Even without specific `[audio]` tags, Phase 3 is designed to proactively enrich your bullet points and notes with relevant details from the audio transcript, providing more context and depth.

*   **Effective Placeholder Use:**
    *   Be strategic with where you place `[slides]` and `[audio]`. Put them in the logical flow of your thoughts.
    *   The system uses timestamps from slides and your notes (if you add them, e.g., "10:15 AM - discussion on X") to help align content.

*   **Review and Refine:**
    *   The AI-generated notes are a powerful draft. Review the final PDF.
    *   Correct any AI misinterpretations or add further personal insights.
    *   The goal is a comprehensive record that combines the factual details (captured by AI) with your unique understanding and perspective.

By leveraging this AI assistant, you can focus more on understanding the content and networking during the conference, knowing that the detailed note-taking is largely automated.

## Troubleshooting

*   **Check the Logs:** If the pipeline fails or produces unexpected results, the first place to look is the `logs/` directory. Each script run generates a timestamped log file for the specific day and phase. Verbose logging (`--verbose`) can provide more details.
*   **API Key:** Ensure your `GOOGLE_API_KEY` in the `.env` file is correct and has the necessary permissions.
*   **Input Files:** Verify that your input files in the `materials/` directory are correctly named and formatted as per the "Input Data Format" section.
*   **Dependencies:** Make sure all dependencies in `requirements.txt` are installed correctly in your virtual environment.
*   **`ffmpeg`:** If Phase 0 (audio transcription) fails, ensure `ffmpeg` is installed and accessible in your system's PATH.

## Contributing

(Placeholder: Add guidelines for contributing to the project if you plan to open it up for collaboration.)

## License

(Placeholder: Specify a license for your project, e.g., MIT, Apache 2.0. If you don't have one yet, you can add "This project is currently unlicensed.")