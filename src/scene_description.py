import json
from dotenv import load_dotenv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

load_dotenv()

# Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "gemini" or "openai"

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("FLASH2.5")

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-kairos")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Initialize providers based on configuration
if LLM_PROVIDER == "gemini":
    from google import genai
    from google.genai import types
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY or FLASH2.5 not found in environment variables.")
elif LLM_PROVIDER == "openai":
    from openai import AzureOpenAI
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in .env file.")
else:
    raise ValueError(f"LLM_PROVIDER must be 'gemini' or 'openai', got '{LLM_PROVIDER}'")

def describe_flash_scene(
                        scene_text: str,
                        client,
                        prompt_path="prompts/flash_scene_prompt_manahil.txt",
                        model = "gemini-2.5-flash", 
                         ) -> str:
    """
    Takes ONE raw scene description (string) and returns
    a concise LLM-generated summary describing:
      - key objects
      - actions
      - spatial relationships
      - temporal relationships
    
    Supports both Gemini and Azure OpenAI providers.
    """

    # Load template prompt from external file
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Insert scene text into {{SCENE_TEXT}} placeholder
    prompt = template.replace("{{SCENE_TEXT}}", scene_text)

    # Use appropriate API based on provider
    if LLM_PROVIDER == "gemini":
        chat = client.chats.create(model=model)
        resp = chat.send_message(prompt)
        return resp.text.strip()
    elif LLM_PROVIDER == "openai":
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise scene descriptions.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                model=model
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e)
            if "DeploymentNotFound" in error_msg or "404" in error_msg:
                raise RuntimeError(
                    f"Azure OpenAI deployment '{model}' not found.\n"
                    f"Please verify:\n"
                    f"  1. The deployment name '{model}' exists in your Azure OpenAI resource\n"
                    f"  2. Your AZURE_OPENAI_DEPLOYMENT in .env matches the actual deployment name\n"
                    f"  3. Your endpoint '{AZURE_OPENAI_ENDPOINT}' is correct\n"
                    f"  4. The deployment is in the same region as your endpoint\n"
                    f"\nTo find your deployment names, check the Azure Portal or run:\n"
                    f"  python test_azure_openai.py"
                ) from e
            raise
    else:
        raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")

def describe_scenes(
    scenes: list,
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
    model: Optional[str] = None,
    prompt_path = "prompts/flash_scene_prompt_manahil.txt",
    debug= False,
    max_workers: int = 1,
    rate_limit_delay: float = 1.0,
):
    """
    Takes a list of scene dictionaries.
    Adds a new key to each: llm_scene_description

    Uses the previously built `format_all_scenes()` to generate
    raw scene descriptions.
    
    Args:
        scenes: List of scene dictionaries
        YOLO_key: Key for YOLO detections
        FLIP_key: Key for frame captions
        ASR_key: Key for audio speech-to-text
        AST_key: Key for audio sound detection
        model: Model name to use (defaults based on provider)
        prompt_path: Path to prompt template file
        debug: Print debug information
        max_workers: Number of parallel threads for API calls (1=sequential, >1=parallel)
        rate_limit_delay: Delay in seconds between API calls (per worker)
    """
    
    # Set default model based on provider if not specified
    if model is None:
        if LLM_PROVIDER == "gemini":
            model = "gemini-2.5-flash"
        elif LLM_PROVIDER == "openai":
            model = AZURE_OPENAI_DEPLOYMENT

    # First format all scenes using your existing system
    formatted_scenes = raw_descriptions(
        scenes,
        YOLO_key=YOLO_key,
        FLIP_key=FLIP_key,
        ASR_key=ASR_key,
        AST_key=AST_key,
    )

    if max_workers == 1:
        # Sequential execution (original behavior)
        updated = []
        
        # Initialize client based on provider
        if LLM_PROVIDER == "gemini":
            client = genai.Client(api_key=GEMINI_API_KEY)
        elif LLM_PROVIDER == "openai":
            client = AzureOpenAI(
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )

        for idx, (scene, formatted_text) in enumerate(zip(scenes, formatted_scenes)):
            summary = describe_flash_scene(formatted_text, 
                                           client, 
                                           prompt_path=prompt_path,
                                           model=model)

            new_scene = dict(scene)
            new_scene["llm_scene_description"] = summary

            updated.append(new_scene)
            if debug: print("Scene",idx, summary)
            time.sleep(rate_limit_delay)  # To avoid rate limits

        return updated
    else:
        # Parallel execution
        return _describe_scenes_parallel(
            scenes=scenes,
            formatted_scenes=formatted_scenes,
            prompt_path=prompt_path,
            model=model,
            debug=debug,
            max_workers=max_workers,
            rate_limit_delay=rate_limit_delay
        )


def _describe_scenes_parallel(
    scenes: list,
    formatted_scenes: list,
    prompt_path: str,
    model: str,
    debug: bool,
    max_workers: int,
    rate_limit_delay: float,
):
    """
    Internal function for parallel scene description.
    Each worker processes scenes independently with its own API client.
    """
    
    def process_single_scene(idx, scene, formatted_text):
        """Process a single scene with its own client instance."""
        try:
            # Each worker gets its own client to avoid threading issues
            if LLM_PROVIDER == "gemini":
                client = genai.Client(api_key=GEMINI_API_KEY)
            elif LLM_PROVIDER == "openai":
                client = AzureOpenAI(
                    api_version=AZURE_OPENAI_API_VERSION,
                    azure_endpoint=AZURE_OPENAI_ENDPOINT,
                    api_key=AZURE_OPENAI_API_KEY,
                )
            
            summary = describe_flash_scene(
                formatted_text, 
                client, 
                prompt_path=prompt_path,
                model=model
            )
            
            new_scene = dict(scene)
            new_scene["llm_scene_description"] = summary
            
            if debug:
                print(f"✓ Scene {idx} completed")
            
            # Rate limiting per worker
            time.sleep(rate_limit_delay)
            
            return (idx, new_scene)
        except Exception as e:
            if debug:
                print(f"✗ Scene {idx} failed: {str(e)}")
            raise
    
    # Submit all tasks to thread pool
    results = [None] * len(scenes)  # Pre-allocate to maintain order
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scenes
        future_to_idx = {}
        for idx, (scene, formatted_text) in enumerate(zip(scenes, formatted_scenes)):
            future = executor.submit(process_single_scene, idx, scene, formatted_text)
            future_to_idx[future] = idx
        
        if debug:
            print(f"\n  ⏳ Processing {len(scenes)} scenes with {max_workers} parallel workers...\n")
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx, new_scene = future.result()
            results[idx] = new_scene
            completed += 1
            if debug:
                print(f"  Progress: {completed}/{len(scenes)} scenes completed")
    
    return results

# ================================================================================================
# SCENE DESCRIPTION FORMATTING

def normalize_bbox(bbox):
    """
    Convert [x1, y1, x2, y2] into raw center + area.
    Useful when we do NOT have frame dimensions.
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    area = w * h

    return x_center, y_center, area

def format_single_description(
    captions: list,
    yolo: dict,
) -> str:
    # Determine number of frames
    frame_count = max(
        len(captions),
        max([int(k) for k in yolo.keys()], default=-1) + 1
    )

    lines = []

    for idx in range(frame_count):
        lines.append(f"Frame {idx}:")

        # ---- Captions ----
        if captions and idx < len(captions):
            cap = captions[idx]
            lines.append(f'  Caption: "{cap}"')

        # ---- YOLO detections ----
        dets = (
            yolo.get(idx)
            if idx in yolo
            else yolo.get(str(idx), [])
        ) or []

        if dets:
            lines.append("  Objects:")

            for det in dets:
                label = det.get("label", "unknown")
                conf = det.get("confidence", 0.0)
                bbox = det.get("bbox", [0, 0, 0, 0])

                x_center, y_center, area = normalize_bbox(bbox)

                obj_str = (
                    f"    - {label} (conf={conf:.2f}), "
                    f"x_center={x_center:.1f}, "
                    f"y_center={y_center:.1f}, "
                    f"area={area:.1f}"
                )
                lines.append(obj_str)
        else:
            lines.append("  Objects: none detected.")

        lines.append("")

    return "\n".join(lines)

def raw_descriptions(
    scenes: list,
    YOLO_key: str = "yolo_detections",
    FLIP_key: str = "frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
) -> list:
    """
    Outer formatter:
      - Reads scenes
      - Skips YOLO or FLIP keys when None
      - Returns a list of scene description strings
    """

    formatted_list = []

    for scene in scenes:
        captions = scene.get(FLIP_key, []) if FLIP_key else []
        yolo = scene.get(YOLO_key, {}) if YOLO_key else {}
        asr = scene.get(ASR_key, "") if ASR_key else ""
        ast = scene.get(AST_key, "") if AST_key else ""

        single_scene_text = format_single_description(
            captions=captions,
            yolo=yolo,
        )

        if asr: single_scene_text += f"\nAudio transcript {asr}\n"
        if ast: single_scene_text += f"\nAudio sounds {ast}\n"

        formatted_list.append(single_scene_text)

    return formatted_list

def test(
    json_path="./captioned_scenes.json",
    YOLO_key="yolo_detections",
    FLIP_key="frame_captions",
    ASR_key: str = "audio_natural",
    AST_key: str = "audio_speech",
):
    """
    Quick test function for raw_descriptions().
    Loads captioned scenes JSON and prints the formatted descriptions.
    """

    # Load scenes
    with open(json_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # Format scenes
    formatted_scenes = raw_descriptions(
        scenes,
        YOLO_key=YOLO_key,
        FLIP_key=FLIP_key,
        ASR_key=ASR_key,
        AST_key=AST_key,
    )

    # Print preview
    print("=" * 60)
    print("Formatted Scene Descriptions")
    print("=" * 60)

    for i, text in enumerate(formatted_scenes):
        print(f"\n--- Scene {i} ---\n")
        print(text)
        print("\n" + "-" * 60)

    return formatted_scenes

# test()
'''
------------------------------------------------------------
--- Scene 8 ---
Frame 0:
  Caption: "a video frame of two men sitting at a table"
  Objects:
    - person (conf=0.94), x_center=165.1, y_center=90.7, area=41058.3
    - person (conf=0.73), x_center=9.9, y_center=142.4, area=1378.2
    - chair (conf=0.61), x_center=50.8, y_center=150.0, area=949.4
Frame 1:
  Caption: "a video frame of pm modi ' s speech in parliament"
  Objects:
    - person (conf=0.95), x_center=163.9, y_center=92.0, area=40143.1
    - person (conf=0.70), x_center=9.3, y_center=144.9, area=1213.2
    - chair (conf=0.60), x_center=50.6, y_center=151.8, area=897.4
Frame 2:
  Caption: "a video frame of pm modi and pm naji"
  Objects:
    - person (conf=0.95), x_center=163.6, y_center=92.3, area=39761.5
    - person (conf=0.69), x_center=9.4, y_center=145.5, area=1197.7
    - chair (conf=0.56), x_center=50.7, y_center=152.2, area=873.2
------------------------------------------------------------
'''
