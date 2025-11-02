"""
Inference and save results to results/[model]/
"""

import argparse
import os
import json
from models import *
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models"))

parser = argparse.ArgumentParser(description='Run AI4ServiceBench')
parser.add_argument("--video_dir", type=str, default="data/EgoLife", help="Root directory of source videos")
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--save_results", type=bool, default=True, help="Save results to a file")
parser.add_argument("--model_path", type=str, required=False, default=None)
args = parser.parse_args()

print(f"Inference Model: {args.model}")


if args.model == "QWen3VL":
    from models.QWen3VL import EvalQWen3VL
    model = EvalQWen3VL(args)
else:
    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")

PROMPT_EXAMPLE = """
You are an AI assistant tasked with analyzing and inferring based on a first-person perspective video recorded through AI glasses. 

Instructions:
1. Monitor the video for event changes.
2. Only output when a new event occurs.
3. For each detected event change, provide:
   - event: the new event type (one of ["social", "housekeeping", "cooking", "shopping", "dining", "party", "arts & craftwork", "leisure", "games", "music & dance", "outing", "setup", "meeting", "commuting"])
   - video_time_str: the timestamp in the video
   - services: at least one suggestion to assist the user in this event

Output format (JSON):
{{
  "event": "<new_event_type>",
  "video_time_str": "<timestamp_read_from_video>",
  "services": ["<service_suggestion1>", "<service_suggestion2>", ...]
}}

Output ONLY when event changes occur. If no event change, do not output anything.
"""

video_files = [os.path.join(root, f)
               for root, _, files in os.walk(args.video_dir)
               for f in files if f.endswith(".mp4")]

results_dir = os.path.join(args.result_dir, args.model)
os.makedirs(results_dir, exist_ok=True)

for video_path in video_files:
    print(f"Processing video: {video_path}")
    segment_outputs = model.inference(video_path, PROMPT_EXAMPLE)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(results_dir, f"{video_name}.jsonl")
    if args.save_results and segment_outputs:
        with open(save_path, "w", encoding="utf-8") as f:
            for line in segment_outputs:
                line = line.strip()
                if line:  
                    f.write(line + "\n")

    print(f"Saved results to: {save_path}" if segment_outputs else "No events detected for this video.")
