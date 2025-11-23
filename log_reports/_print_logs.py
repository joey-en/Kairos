import json
import pandas as pd
import glob
import os

LOG_DIR = "./logs"
OUTPUT_DIR = "./log_reports"
OUTPUT_MD = os.path.join(OUTPUT_DIR, "all_logs_report.md")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

STEP_KEYS = {
    "get_scene_list": ("video_length", "PySceneDetect*"),
    "ast_timings": ("video_length", "AST sound descriptions*"),
    "asr_timings": ("video_length", "ASR speech transcription*"),
    "save_clips": ("scene_number", "Masked clips saving"),
    "sample_frames": ("scene_number", "Frame sampling"),
    "caption_frames": ("scene_number", "BLIP caption"),
    "detect_object_yolo": ("scene_number", "YOLO detection"),
    "describe_scenes": ("scene_number", "BLIP + YOLO + AST + ASR into LLM"),
}

METRIC_COLUMNS = [
    "wall_time_sec",
    "cpu_time_sec",
    "ram_used_MB",
    "io_read_MB",
    "io_write_MB"
]

def safe_div(x, d):
    return x / d if d not in [0, None] else x

markdown_sections = []

json_files = glob.glob(os.path.join(LOG_DIR, "*.json"))

for file_path in json_files:
    # Fix for UnicodeDecodeError:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        document = json.load(f)

    video_path = document.get("video_path", "unknown")
    video_title = os.path.basename(video_path)
    scene_count = document.get("scene_number", 1)

    rows = []

    for step_key, (divisor_key, friendly_name) in STEP_KEYS.items():
        step_data = document.get("steps", {}).get(step_key, {})
        row = {"step": friendly_name}  # Use friendly name here

        divisor_value = document.get(divisor_key, 1)

        for metric in METRIC_COLUMNS:
            raw_value = step_data.get(metric, 0)

            row[metric] = raw_value / divisor_value if divisor_value else raw_value

            # Special rule for describe_scenes because of delay 
            if step_key == "describe_scenes" and metric == "wall_time_sec":
                row[metric] -= 5 # time.sleep(5) for API

            if step_key in ["get_scene_list", "ast_timings", "asr_timings"]:
                row[metric] *= 60 

        rows.append(row)


    df = pd.DataFrame(rows).applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)

    # CSV name
    base_name = os.path.splitext(video_title)[0].replace(" ", "_")
    csv_path = os.path.join(OUTPUT_DIR, f"{base_name}.csv")

    df.to_csv(csv_path, index=False)

    # Markdown section
    md = f"## {video_title}\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n"

    video_length = document.get("video_length", 1)
    total_sec = document.get("total_process_sec", 1)

    k = (total_sec / video_length) if video_length > 0 else 0

    md += (
        f"**Footnote:**  \n"
        f"`total_process_sec` is **{k:.2f}× longer** than `video_length` of {document.get("video_length", "`error: video_length not found`")}s.  \n"
        f"**{scene_count} scenes** were detected in `{video_path}` \n"
        f"**`get_scene_list`, `ast_timings`, and `asr_timings` are measured per minute of video, whereas the remaining processes are measured per scenes. \n"
    )

    markdown_sections.append(md)

# Write all markdown tables
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write("# Processing Logs Summary\n\n")
    for section in markdown_sections:
        f.write(section)

print("Done! Markdown + CSVs generated in ./log_reports")
