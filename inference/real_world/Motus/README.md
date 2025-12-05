# Real-World Inference (No Environment)

This folder provides a minimal example to run Motus inference on a single image **without any robot environment**.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Files](#files)
- [Usage](#usage)
  - [With Pre-encoded T5 (Recommended)](#with-pre-encoded-t5-recommended)
  - [With On-the-fly T5 Encoding](#with-on-the-fly-t5-encoding)
- [Output](#output)
- [Notes](#notes)

---

## Overview

Two inference modes are supported:

| Mode | VRAM | Description |
|------|------|-------------|
| **With pre-encoded T5** | ~24 GB | Use pre-encoded T5 embeddings (recommended for deployment) |
| **With on-the-fly T5** | ~41 GB | Encode instruction text at runtime (requires larger GPU) |

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **Motus Checkpoint** | Directory containing `mp_rank_00_model_states.pt` |
| **WAN Path** | Path to WAN models (for T5 encoder and VAE) |
| **Input Image** | Single RGB image (will be resized to model input size) |
| **Instruction** | Text instruction describing the task |

---

## Files

| File | Description |
|------|-------------|
| `inference_example.py` | Main inference script (CLI + Python API examples) |
| `encode_t5_instruction.py` | Utility to encode text instructions to T5 embeddings |
| `utils/robotwin.yml` | Model configuration for RoboTwin checkpoint |

---

## Usage

### With Pre-encoded T5 (Recommended)

This mode uses ~24GB VRAM and is recommended for deployment on RTX 5090 or similar GPUs.

**Step 1:** Encode the instruction to T5 embeddings (do this once per instruction):

```bash
python inference/real_world/Motus/encode_t5_instruction.py \
  --instruction "pick up the cube and place it on the right" \
  --output t5_embed.pt \
  --wan_path /path/to/pretrained_models
```

**Step 2:** Run inference with pre-encoded embeddings:

```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/robotwin.yml \
  --ckpt_dir ./pretrained_models/Motus_robotwin2 \
  --wan_path /path/to/pretrained_models \
  --image /path/to/input_frame.png \
  --instruction "pick up the cube and place it on the right" \
  --t5_embeds t5_embed.pt \
  --output result.png
```

### With On-the-fly T5 Encoding

This mode uses ~41GB VRAM and requires A100 (40GB) or larger.

```bash
python inference/real_world/Motus/inference_example.py \
  --model_config inference/real_world/Motus/utils/robotwin.yml \
  --ckpt_dir ./pretrained_models/Motus_robotwin2 \
  --wan_path /path/to/pretrained_models \
  --image /path/to/input_frame.png \
  --instruction "pick up the cube and place it on the right" \
  --use_t5 \
  --output result.png
```

---

## Output

| Output | Description |
|--------|-------------|
| `result.png` | Image grid: condition frame + predicted future frames |
| Console | Predicted action chunk with shape `(action_chunk_size, action_dim)` |

**Example console output:**

```
Predicted actions shape: (48, 14)
First 3 actions:
[[ 0.012  0.003 -0.001  0.002  0.001  0.000  0.045 ...]
 [ 0.015  0.004 -0.002  0.003  0.001  0.001  0.048 ...]
 [ 0.018  0.005 -0.003  0.004  0.002  0.001  0.051 ...]]
```

---

## Notes

- `--ckpt_dir` should point to the Motus checkpoint directory (containing `mp_rank_00_model_states.pt`)
- `--wan_path` is the base path for WAN models, used to locate T5 weights and VAE
- The `--instruction` argument is always required (for VLM processing even when using pre-encoded T5)
- For batch processing multiple instructions, pre-encode all T5 embeddings first to avoid repeated model loading
