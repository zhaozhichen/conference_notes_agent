import re
from pathlib import Path
import json
from datetime import datetime, timedelta


def parse_timeblock(time_str):
    # Expects format like "01:15 PM"
    t = datetime.strptime(time_str, "%I:%M %p")
    return t.hour, t.minute


def segment_transcript(transcript_path, sessions, output_dir):
    """
    Split the transcript into session files based on session start/end times.
    transcript_path: Path to the master transcript file.
    sessions: list of dicts with 'title', 'start', 'end'.
    output_dir: Path to write session transcript files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find all timestamps in transcript (e.g. [10:15:32])
    time_re = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]")
    line_times = []
    for i, line in enumerate(lines):
        m = time_re.search(line)
        if m:
            h, m_, s = map(int, m.groups())
            line_times.append((i, h, m_, s))

    # For each session, collect lines within its time window
    for idx, sess in enumerate(sessions):
        if not sess["start"] or not sess["end"]:
            continue
        sh, sm = parse_timeblock(sess["start"])
        eh, em = parse_timeblock(sess["end"])
        session_lines = []
        in_window = False
        for i, line in enumerate(lines):
            m = time_re.search(line)
            if m:
                h, m_, s = map(int, m.groups())
                t = h * 60 + m_  # minutes since midnight
                if t >= sh * 60 + sm and t <= eh * 60 + em:
                    in_window = True
                else:
                    in_window = False
            if in_window:
                session_lines.append(line)
        if session_lines:
            fname = f"session_{idx+1:02d}_{sess['title'].replace(' ', '_')}.txt"
            with open(output_dir / fname, "w", encoding="utf-8") as f:
                f.writelines(session_lines) 