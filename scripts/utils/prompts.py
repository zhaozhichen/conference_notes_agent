OCR_PROMPT = (
    "You are provided with a photograph of a presentation slide. "
    "Extract all visible text content. If tables are present, represent them in a text-based format (e.g., Markdown). "
    "If diagrams or flowcharts are present, describe their components and logical relationships in text. "
    "Return the extracted information clearly. If specific elements (text, tables, diagrams) are not present, indicate that or omit the section."
)

PHASE2_CONTEXT_PROMPT = """
You are an AI assistant tasked with contextualizing conference materials.
You will be given:
1. A conference agenda.
2. A user's manual notes taken during the conference.
3. A list of extracted slide contents, each with a timestamp and original filename.

Your tasks are to:
A. Identify which conference sessions the user likely attended based on their manual notes and the agenda.
B. For each slide entry in the extracted slide contents, determine which of the attended sessions it belongs to by comparing the slide's timestamp with the session times in the agenda. Assign a session title to each slide.

Provide your output in JSON format with the following top-level keys:
- "attended_sessions": A list of strings, where each string is the exact title of an attended session from the agenda.
- "slide_to_session_mapping": A list of objects. Each object should have:
    - "slide_identifier": String, the original unique identifier for the slide (e.g., its timestamp and filename from the input like "[YYYY-MM-DD HH:MM:SS (Source)] Slide: FILENAME.jpg").
    - "determined_session_title": String, the exact title of the session (from the agenda and your list of attended sessions) to which this slide has been mapped. If a slide cannot be mapped to an attended session, this can be null or an empty string.

Here is the data:

--- AGENDA START ---
{agenda_string}
--- AGENDA END ---

--- USER NOTES START ---
{user_notes_string}
--- USER NOTES END ---

--- EXTRACTED SLIDE CONTENTS START ---
{extracted_slides_string}
--- EXTRACTED SLIDE CONTENTS END ---

Please generate the JSON output.
"""

PHASE2_MERGE_PROMPT = """
You are an AI assistant. Your task is to merge session titles into a document containing extracted slide content.
You will be given two pieces of text:
1.  "EXTRACTED SLIDE CONTENT": This text contains multiple slide entries. Each slide entry starts with a header line like "[TIMESTAMP (Source)] Slide: FILENAME.jpg", followed by its extracted text, tables, and diagram descriptions. Entries are separated by "---".
2.  "SLIDE-TO-SESSION MAPPING JSON": This is a JSON array where each object maps a "slide_identifier" (which matches the header line format from the "EXTRACTED SLIDE CONTENT") to a "determined_session_title".

Your goal is to produce an updated version of the "EXTRACTED SLIDE CONTENT". For each slide entry in the original "EXTRACTED SLIDE CONTENT":
- Locate its corresponding "determined_session_title" from the "SLIDE-TO-SESSION MAPPING JSON" using the "slide_identifier".
- Insert a new line immediately after the slide's header line, formatted as: "Session Title: <The Determined Session Title>"
- If a slide_identifier from the "EXTRACTED SLIDE CONTENT" is not found in the "SLIDE-TO-SESSION MAPPING JSON", or if its "determined_session_title" is null or an empty string, insert "Session Title: Not Determined".
- Preserve all other original content and formatting from the "EXTRACTED SLIDE CONTENT".

Here is the "EXTRACTED SLIDE CONTENT":
--- EXTRACTED SLIDE CONTENT START ---
{extracted_slides_string}
--- EXTRACTED SLIDE CONTENT END ---

Here is the "SLIDE-TO-SESSION MAPPING JSON":
--- SLIDE-TO-SESSION MAPPING JSON START ---
{slide_session_mapping_json_string}
--- SLIDE-TO-SESSION MAPPING JSON END ---

Please output only the complete, updated "EXTRACTED SLIDE CONTENT" with the session titles inserted. Do not add any extra explanations or commentary outside of the modified content itself.
"""

PHASE2_AUDIO_SEGMENTATION_PROMPT = """
You are an AI assistant tasked with segmenting a master audio transcript into individual session transcripts based on a conference agenda and a list of attended session titles.

You will be given:
1.  "CONFERENCE AGENDA": A string containing the full conference agenda, with session titles, start times, and potentially end times.
2.  "ATTENDED SESSION TITLES": A list of exact session titles that the user attended.
3.  "MASTER AUDIO TRANSCRIPT": A single string containing the complete transcription of all audio, potentially with timestamps (e.g., "[HH:MM:SS]" or similar) or other markers.

Your tasks are to:
A. For each title in the "ATTENDED SESSION TITLES" list:
    1. Locate the corresponding session details (start time, end time if available) in the "CONFERENCE AGENDA".
    2. Identify the segment of the "MASTER AUDIO TRANSCRIPT" that corresponds to this attended session. Use session start times and the sequence of sessions in the agenda to determine approximate boundaries. If the transcript has timestamps, use them for more precise segmentation.
    3. Extract this segment.

B. Present the output as a single continuous text document. For each attended session, clearly indicate the session title, followed by its corresponding transcript segment. Use a clear separator (e.g., "--- SESSION START: [Session Title] ---") before each segment and "--- SESSION END: [Session Title] ---" after each segment.

Example format for each session in the output:

--- SESSION START: Keynote: The Future of AI ---
Speaker: Welcome everyone to this exciting keynote... 
... (transcript content for this session) ...
...Thank you very much.
--- SESSION END: Keynote: The Future of AI ---

--- SESSION START: Next Attended Session Title ---
... (transcript content for this session) ...
--- SESSION END: Next Attended Session Title ---

If a session from the "ATTENDED SESSION TITLES" cannot be clearly segmented from the "MASTER AUDIO TRANSCRIPT" (e.g., due to unclear boundaries or missing content), include a note like "--- SESSION START: [Session Title] ---\\n[Content for this session could not be clearly segmented.]\\n--- SESSION END: [Session Title] ---". Ensure all attended sessions are addressed.

Here is the data:

--- CONFERENCE AGENDA START ---
{agenda_string}
--- CONFERENCE AGENDA END ---

--- ATTENDED SESSION TITLES START ---
{attended_session_titles_string} 
--- ATTENDED SESSION TITLES END ---

--- MASTER AUDIO TRANSCRIPT START ---
{master_transcript_string}
--- MASTER AUDIO TRANSCRIPT END ---

Please generate the complete, single text output containing all segmented session transcripts formatted as described.
"""

PHASE3_SYNTHESIS_PROMPT = """
You are an AI conference note-taking assistant. Your goal is to synthesize comprehensive, enriched notes by integrating information from three sources: user's manual notes, extracted slide content, and segmented audio transcripts. The user's manual notes should serve as the primary structural and thematic guide.

You will be given:
1.  "USER MANUAL NOTES": The user's raw notes, which may include personal observations, keywords, questions, and placeholders like `[slides]`, `[audio]`, and `[todo]`. This document dictates the overall flow and areas of focus.
2.  "MAPPED SLIDE CONTENT": This contains text, table representations, and diagram descriptions extracted from presentation slides. Each slide entry is identified by a header (e.g., "[TIMESTAMP (Source)] Slide: FILENAME.jpg") and includes its "Session Title".
3.  "SEGMENTED AUDIO TRANSCRIPTS": This is a text document containing audio transcripts segmented by session, with headers like "--- SESSION START: [Session Title] ---" and "--- SESSION END: [Session Title] ---" demarking each session's transcript.

Your tasks are to:
A.  **Understand the User's Focus**: Use the "USER MANUAL NOTES" as the backbone. The topics, questions, and structure mentioned by the user are paramount.
B.  **Synthesize and Enrich**:
    When integrating information, give particular consideration to the "SEGMENTED AUDIO TRANSCRIPTS" as they may contain subtle but crucial details, clarifications, or discussions not present in the slides or manual jottings. Your goal is to capture the full depth of information available.
    1.  For each section or point in the "USER MANUAL NOTES":
        *   **If a `[slides]` placeholder is present**: Locate the most relevant slide(s) from "MAPPED SLIDE CONTENT" based on the context (surrounding text in manual notes, session title if implied, keywords, timestamps). Integrate key information from the slide (text, bullet points, table summaries, diagram descriptions) to elaborate on the user's note. After inserting the slide content, add a Markdown image reference like "![[cropped_slides/FILENAME.jpg]]" (extract FILENAME.jpg from the slide header).
        *   **If an `[audio]` placeholder is present**: Find the relevant session in "SEGMENTED AUDIO TRANSCRIPTS" based on context from the manual notes. Then, locate the specific part of that session's transcript that addresses the user's point. Insert a pertinent verbatim quote (using Markdown blockquotes) or a concise summary of the speaker's explanation. Actively look for details in the audio that provide more depth or context than what might be in the slides or manual notes for this point.
        *   **Beyond explicit placeholders (Proactive Audio Enrichment)**: For *any* point in the "USER MANUAL NOTES", especially for short or succinct bullet points, actively search the corresponding session's "SEGMENTED AUDIO TRANSCRIPTS" for explanations, examples, or discussions that elaborate on that point. Integrate these insights as concise summaries or relevant verbatim quotes (using Markdown blockquotes) to add depth. This is crucial even if no `[audio]` tag is present. Also, integrate relevant details, tables, or diagram descriptions from "MAPPED SLIDE CONTENT" where appropriate. Aim to make the notes comprehensive, prioritizing detailed insights from audio when available to expand on brief notes.
    2.  **Maintain Coherence**: Ensure the integrated information flows logically with the user's original notes. The output should feel like a single, well-structured document, not just a list of resolved placeholders.
    3.  **Formatting**: Use Markdown for formatting. Present tables from slides as Markdown tables. Use bullet points or numbered lists where appropriate for clarity.
C.  **Extract To-Do Items**:
    *   Identify all `[todo]` placeholders in the "USER MANUAL NOTES".
    *   Extract the full text of each to-do item (e.g., "[todo] Research Ashby's Law" becomes "Research Ashby's Law").
    *   Collect these into a simple list of strings.
D.  **Produce Output in Two Distinct Blocks**: Structure your entire response as follows, with no other text before or after these blocks:

ENRICHED_MARKDOWN_NOTES_START
(The complete, synthesized notes in Markdown format go here. All `[slides]` and `[audio]` placeholders should be resolved and integrated. `[todo]` placeholders should be removed from this final notes text.)
ENRICHED_MARKDOWN_NOTES_END

TODO_ITEMS_START
(Each extracted to-do item on a new line. For example:
Research Ashby's Law
Follow up on X
)
TODO_ITEMS_END

Here is the data:

--- USER MANUAL NOTES START ---
{user_manual_notes_string}
--- USER MANUAL NOTES END ---

--- MAPPED SLIDE CONTENT START ---
{mapped_slide_content_string}
--- MAPPED SLIDE CONTENT END ---

--- SEGMENTED AUDIO TRANSCRIPTS START ---
{segmented_audio_transcripts_string}
--- SEGMENTED AUDIO TRANSCRIPTS END ---

Please generate the output using the two distinct blocks as specified above.
"""