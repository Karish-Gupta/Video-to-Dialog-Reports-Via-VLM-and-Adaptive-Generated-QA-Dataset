from yt_dlp import YoutubeDL
import requests
import json
import os
import re

def _safe_filename(s):
    s = re.sub(r'[\\/*?:"<>|]', "", s)
    s = s.strip().replace(" ", "_")
    return s[:200]

def extract_subtitles(url, save_dir=None):
    ydl_opts = {
        'skip_download': True,
        'cookiefile': r'cookies.txt',   # <-- use exported cookies
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitlesformat': 'vtt',
        'subtitleslangs': ['en'],  
        'quiet': True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get('subtitles') or info.get('automatic_captions')
        if not subs:
            print("No subtitles found.")
            return None

        # Pick English subtitles if available
        sub_url = subs.get('en')[0]['url']
        print(f"Subtitle URL: {sub_url}")

        # Download the subtitle file and print it
        import requests
        r = requests.get(sub_url)
        vtt_data = r.text
        
        # If YouTube returned the json3 auto-caption format, parse it to plain text.
        if vtt_data.lstrip().startswith("{"):
            try:
                j = json.loads(vtt_data)
                parts = []
                for ev in j.get("events", []):
                    for seg in ev.get("segs", []):
                        # json3 uses "utf8" for text segments
                        txt = seg.get("utf8") or seg.get("utf-8") or seg.get("text")
                        if txt and txt.strip():
                            parts.append(txt.strip())
                text = " ".join(parts)
                print(text[:1000])  # preview first 1000 chars
            except Exception as e:
                print("Failed to parse json3 subtitles:", e)
                text = vtt_data
        else:
            # Likely VTT or plain text
            text = vtt_data

        vid = info.get("id", "video")
        title = info.get("title", vid)
        fname = _safe_filename(f"{vid}_{title}") + ".txt"
        out_dir = save_dir or os.path.dirname(__file__) or "."
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Saved subtitles to: {out_path}")
        return out_path


extract_subtitles("https://www.youtube.com/watch?v=OuIrm8p_FYk")
