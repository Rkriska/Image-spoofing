---
title: Image Spoofing Detection
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Image Spoofing Detection — Find IT! 2026

Computer vision project for detecting face spoofing attacks in face-recognition systems.  
The deployed demo classifies face images into six classes, then simplifies the result into a practical **Real Face / Spoof Detected** workflow.

This project was developed for **Data Analytics — Find IT! 2026** by team **bismillah kasih keras**.

## Project Objective

Face recognition systems are vulnerable to spoofing attacks such as printed photos, replayed images on screens, masks, mannequins, or other visual impersonation methods. This project builds an anti-spoofing classifier to detect whether a face image comes from a real person or a spoofing medium.

```txt
Input  : face image
Output : fake_mannequin / fake_mask / fake_printed / fake_screen / fake_unknown / realperson
Metric : Macro F1-Score