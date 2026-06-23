---
title: Natanyx Redrob Ranker
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Natanyx Redrob Ranker

Sandbox demo for the Redrob hackathon candidate ranker.

The app runs the same deterministic feature-engineered scoring logic as the submission code. It includes the public sample candidates and also accepts an uploaded JSON or JSONL file with up to 100 candidate records.

The sandbox does not use network calls, hosted LLM APIs, or GPU inference during ranking.
