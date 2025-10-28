from yt_dlp import YoutubeDL
import requests
import re
import json
import os


def _safe_filename(s: str) -> str:
    """Sanitize filenames for filesystem safety."""
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    s = s.strip().replace(" ", "_")
    return s[:200]


def parse_vtt(vtt_text: str):
    """Parse WebVTT subtitle text into structured data."""
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})")
    entries = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        match = pattern.match(lines[i])
        if match:
            start, end = match.groups()
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not pattern.match(lines[i]):
                text_lines.append(lines[i].strip())
                i += 1
            if text_lines:
                entries.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(text_lines)
                })
        i += 1
    return entries


def parse_json3(json_text: str):
    """Parse YouTube JSON3 auto-captions (used when .vtt isn't available)."""
    data = json.loads(json_text)
    captions = []
    for ev in data.get("events", []):
        start = ev.get("tStartMs", 0) / 1000.0
        duration = ev.get("dDurationMs", 0) / 1000.0
        text = " ".join(
            seg.get("utf8", "").strip() for seg in ev.get("segs", []) if seg.get("utf8")
        )
        if text:
            captions.append({
                "start": f"{start:.3f}",
                "end": f"{start + duration:.3f}",
                "text": text
            })
    return captions


def extract_subtitles_with_timestamps(url: str, save_dir: str = None):
    """Extract subtitles (VTT or JSON3) and save structured timestamped data."""
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get('subtitles') or info.get('automatic_captions')
        if not subs or 'en' not in subs:
            print("No English subtitles found.")
            return None

        sub_url = subs['en'][0]['url']
        print(f"Downloading subtitles from: {sub_url}")
        response = requests.get(sub_url)
        response.raise_for_status()
        subtitle_data = response.text.strip()

        # Detect format (JSON or VTT)
        if subtitle_data.startswith("{"):
            print("Detected JSON3 format — parsing as JSON.")
            parsed_entries = parse_json3(subtitle_data)
        else:
            print("Detected VTT format — parsing as VTT.")
            parsed_entries = parse_vtt(subtitle_data)

        print(f"Extracted {len(parsed_entries)} subtitle entries.")

        # Save result
        vid = info.get("id", "video")
        title = info.get("title", vid)
        base_name = _safe_filename(f"{vid}_{title}")
        save_dir = save_dir or os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{base_name}_subtitles.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_entries, f, indent=2, ensure_ascii=False)

        print(f"Saved subtitles with timestamps to:\n{out_path}")
        return out_path


if __name__ == "__main__":
    extract_subtitles_with_timestamps("https://www.youtube.com/watch?v=OuIrm8p_FYk", save_dir="outputs")
