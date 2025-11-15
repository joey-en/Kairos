---

# 🚀 Kairos — Multimodal Video Understanding Pipeline

Kairos is a research-grade pipeline that performs **scene detection**, **frame captioning**, **speech recognition**, and **natural audio tagging** to produce a **complete multimodal caption** for each scene in a video.

The repository also includes **experimental motion-detection research** using MOG2, KNN, and simple frame differencing. These experiments live in isolated folders and do **not** affect the main pipeline.

---

# 📂 Repository Structure (Final)

```
Kairos/
│
├── main.py                         # Main multimodal pipeline (production)
│
├── src/
│   ├── scene_cutting.py            # PySceneDetect wrapper
│   ├── frame_sampling.py           # Extract frames from scenes
│   ├── frame_captioning_blip.py    # BLIP lightweight captioner
│   ├── frame_captioning_heavy.py   # (Optional) heavy captioner
│   ├── audio_asr.py                # Whisper ASR + RMS + noise reduction
│   ├── audio_natural.py            # AST natural audio tagging
│   ├── debug_utils.py              # Scene visualization helpers
│   └── __init__.py
│
├── output/                         # Auto-generated results
│   ├── frames/
│   ├── audio/
│   ├── captions/
│   └── ...
│
├── Videos/                         # Input videos
│
├── test_frame_differencing/        # Algorithm-level experiments
│   ├── simple_frame_differencing.py
│   ├── MOG2_frame_differencing.py
│   ├── KNN_frame_differencing.py
│   ├── utils.py
│   └── test_frame_differencing.py
│
├── test_frame_differencing_pipeline/
│   ├── main.py                     # Pipeline test using MOG2 per scene
│   ├── scene_detector.py           # PySceneDetect wrapper (experimental)
│   ├── blip_captioner.py           # Captions frames in this pipeline
│   ├── mog2_frame_differencing.py  # MOG2 scene-level scoring
│   ├── metrics.py                  # Profiling utils
│   ├── utils.py
│   └── output/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 🧠 Main Pipeline Overview (`main.py`)

The production pipeline in the **root folder** performs:

### **1. Scene Detection**

* Built on **PySceneDetect**
* Detects high-level scene boundaries
* Produces `start_sec` and `end_sec` for each scene

### **2. Frame Sampling**

* 2 frames per scene (configurable)
* Saved to `output/frames/`

### **3. Frame Captioning (BLIP)**

* Lightweight BLIP captioning
* Produces a caption for each frame
* Combined into **scene-level visual caption**
* Saved to `output/captions/scene_X_blip.txt`

### **4. Audio Extraction**

* Extracts each scene’s audio (`.wav`)
* Saved to `output/audio/scene_X.wav`

### **5. Whisper ASR with RMS + Noise Reduction**

* RMS-based silence detection (explained later)
* Spectral noise reduction
* Whisper (small/medium)
* Saved to `scene_X_asr.txt`

### **6. Natural Audio Tagging (AST)**

* Detects environmental sounds:

  * music
  * doors
  * footsteps
  * wind
  * ambience
* Saved to `scene_X_audio_labels.json`

### **7. Final Scene Caption**

Each scene ends with a **complete multimodal caption**:

```
[BLIP frame caption] 
+ [Speech transcript]
+ [Natural audio labels]
```

Saved as:

```
output/captions/scene_X_full_caption.txt
```

---

# 🔉 Audio Modules Explained (src/audio_asr.py & src/audio_natural.py)

## 🎤 `audio_asr.py` — Whisper ASR + RMS Silence Detection

Whisper often **hallucinates speech** when the audio is silent.

To fix this, we added:

### ✔ RMS (Root Mean Square) Energy Detector

**Why RMS?**

* Measures actual signal energy
* If clip energy < threshold → **return `"silent"`**
* Prevents hallucinated outputs like *“Thanks for watching!”*

### ✔ Noise Reduction

* Simple spectral-gating
* Removes background hum and noise

### ✔ Whisper Transcription

* Supports `"small"` or `"medium"`
* `"medium"` is more accurate
* `"small"` is faster and still usable

---

## 🎧 `audio_natural.py` — Natural Audio Event Detection (AST)

Uses HuggingFace’s **Audio Spectrogram Transformer (AST)**.

### Detects real-world sounds:

* footsteps
* rain
* music
* chatter
* explosions
* machinery
* environmental ambience

### Why AST?

* Whisper only transcribes **speech**
* Videos contain important **non-speech context**
* AST enriches scene understanding dramatically

---

# 🧪 Experimental Motion Detection (Not in Main Pipeline)

Motion detection was **NOT** added to the main pipeline yet.

Instead, there are two experimental folders:

---

# 📁 `test_frame_differencing/`

### Purpose: **Pure algorithm research**

This folder tests **three** classic motion detection techniques:

---

## 🟦 1. Simple Frame Differencing

A baseline method:

### How it works

* Convert frames to grayscale
* Compute `abs(curr - prev)`
* Apply threshold

### Pros

* Very fast
* Zero dependencies
* CPU-only

### Cons

* Very noisy
* Lighting changes break it
* Not robust enough for production

---

## 🟩 2. MOG2 — Mixture of Gaussians

Best performer.

### How it works

* Learns statistical background model
* Foreground = anything that deviates from background

### Pros

* Stable
* Handles lighting changes
* Very good masks

### Cons

* Slightly slower

📌 **Used in pipeline-level test.**

---

## 🟥 3. KNN Background Subtraction

ML-based background model.

### Pros

* Smooth masks
* Great when background changes

### Cons

* Slowest
* Higher memory usage

---

### 🧾 Performance Comparison (from your real results)

```
⏱ Processing Time:
  Simple: 3.43s
  MOG2:   10.17s
  KNN:    31.17s

🚀 Speed (fps):
  Simple: 254 fps
  MOG2:    85 fps
  KNN:     28 fps

💾 RAM:
  Simple: 577 MB
  MOG2:   1331 MB
  KNN:    2101 MB

🎯 Motion Quality (avg pixels changed):
  Simple: 79k
  MOG2:   362k
  KNN:    400k
```

### Recommendation

📌 **Use MOG2** if motion detection is added in future.

---

# 📁 `test_frame_differencing_pipeline/`

### Purpose: **Integration test with scenes**

This folder performs a full test pipeline:

1. PySceneDetect → detect scenes
2. Sample frames per scene
3. Apply **MOG2 motion detection**
4. Convert motion mask → **scene-level motion score**

### Output example:

```
output/
   scene_0_motion.json
   scene_1_motion.json
   ...
```

This shows how motion detection *could* be integrated into the main pipeline later.

The **production pipeline does NOT use motion detection** yet.

---

# 🔥 Why Frame Differencing Is Not in Main Pipeline

* We want the team to evaluate results from
  `test_frame_differencing_pipeline/`
  before deciding if it's stable enough.
* Motion detection is still experimental.
* PySceneDetect already segments scenes reliably.

---

# 📌 Summary for the Team

### **Main pipeline uses:**

✔ PySceneDetect
✔ BLIP captioning
✔ Whisper ASR with RMS + noise reduction
✔ AST natural audio tagging

### **Experiments (optional review):**

* Simple differencing
* MOG2
* KNN
* Scene-level MOG2 motion scoring pipeline

---

# 🎯 Future Improvements

* Add MOG2 motion scoring into main pipeline (if approved)
* Add CLIP / SigLIP global caption refinement
* Add LLM fusion module for unified scene narrative

---
