# Parallel Processing Guide

## Overview

The video processing pipeline has been optimized with **3 stages of parallelization** to significantly speed up processing while respecting API rate limits.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: Scene Detection (Sequential)                      │
│  • Scene detection from video                               │
│  • Save scene clips                                         │
│  • Sample frames from each scene                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Vision + Audio Processing (4 Parallel Tasks)      │
│                                                             │
│  Task 1: BLIP Frame Captioning                              │
│    └─ Processes scenes in parallel (configurable workers)  │
│                                                             │
│  Task 2: YOLO Object Detection                              │
│    └─ Processes scenes in parallel (configurable workers)   │
│                                                             │
│  Task 3: AST Sound Extraction                               │
│    └─ Processes scenes in parallel (configurable workers)    │
│                                                             │
│  Task 4: ASR Speech Recognition                             │
│    └─ Processes scenes in parallel (configurable workers)    │
│                                                             │
│  All 4 tasks run simultaneously, each with internal         │
│  scene-level parallelism for maximum throughput             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Scene Description (Configurable Parallel Workers) │
│  • Gemini API calls to describe each scene                  │
│  • Parallelism controlled by DESCRIBE_SCENES_MAX_WORKERS    │
│  • Rate limiting controlled by DESCRIBE_SCENES_RATE_LIMIT   │
└─────────────────────────────────────────────────────────────┘
```
