import re
from datetime import datetime, timedelta
from pathlib import Path
# Note: json import might be needed if caching is re-introduced or if output format is JSON.
# For now, it's not strictly necessary if we remove caching.
# import json 

# Caching for local file parsing is removed for simplicity.
# If parsing `agenda.txt` becomes a performance bottleneck, caching can be re-added.
# Example of a configurable cache path if needed in the future:
# from .config import WORKING_DATA_DIR 
# AGENDA_PARSED_CACHE_FILE = WORKING_DATA_DIR / "agenda_parsed_cache.json"

def parse_local_agenda(agenda_path: Path):
    """Parse a local agenda.txt file and return a list of sessions with title, start, end, and day."""
    with open(agenda_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    sessions = []
    day = None
    for i, line in enumerate(lines):
        m_day = re.match(r"(Monday|Tuesday|Wednesday|Thursday|Friday), (\d{1,2} \w+)", line)
        if m_day:
            day = line
            continue
        m = re.match(r"(\d{2}:\d{2} [AP]M) \(PDT\)", line)
        if m:
            start = m.group(1)
            titles = []
            j = i + 1
            while j < len(lines) and not re.match(r"\d{2}:\d{2} [AP]M", lines[j]):
                if re.match(r"(Monday|Tuesday|Wednesday|Thursday|Friday),", lines[j]):
                    break
                if not lines[j].startswith("Break") and not lines[j].startswith("Lunch") and not lines[j].startswith("Expo") and not lines[j].startswith("Registration"):
                    titles.append(lines[j])
                j += 1
            for title in titles:
                sessions.append({
                    "day": day,
                    "start": start,
                    "title": title,
                })
    # Infer end times
    for idx, sess in enumerate(sessions):
        # Find next session on same day
        next_idx = idx + 1
        while next_idx < len(sessions) and sessions[next_idx]["day"] != sess["day"]:
            next_idx += 1
        if next_idx < len(sessions) and sessions[next_idx]["day"] == sess["day"]:
            sess["end"] = sessions[next_idx]["start"]
        else:
            # Default: 1 hour after start
            t = datetime.strptime(sess["start"], "%H:%M %p")
            t_end = t + timedelta(hours=1)
            sess["end"] = t_end.strftime("%I:%M %p").lstrip("0")
    return sessions 