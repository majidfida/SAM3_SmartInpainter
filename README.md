```markdown
# SAM3 Smart Inpainter for ComfyUI

> 🎯 One node to find faces (or any object) and repaint them — using text prompts.  
> ✅ Tested with **SAM3 only** (`sam3.pt`)

---

## ✨ What This Node Does

This node combines **segmentation** + **inpainting** in one step:

1. You type a prompt like `"face"`, `"eyes"`, `"scarf"`
2. The node finds that area using the **SAM3** AI model
3. It repaints *only that area* using your diffusion model
4. Blends the result back into your original image

✅ Great for: face detail, fixing eyes, changing accessories, removing small objects  
✅ Works with: any ComfyUI text-to-image workflow  
✅ Model supported: **SAM3** (`sam3.pt` from HuggingFace)

---

## 📦 What You Must Install

### Step 1: Install a SAM3 Wrapper (Required)

This node needs **one** of these custom nodes to provide the `sam3` Python package:

#### Option A (Recommended): ComfyUI-Easy-Sam3
```bash
cd custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Sam3.git
```

#### Option B: ComfyUI-RMBG
```bash
cd custom_nodes
git clone https://github.com/1038lab/ComfyUI-RMBG.git
```

### Step 2: Install Python Requirements

A `requirements.txt` file is included. Install it using the method that matches your ComfyUI setup:

---

## 🔧 Installation Guide

### For ComfyUI Portable (Windows `python_embeded`)

```bash
# 1. Go to ComfyUI root folder
cd F:\ComfyUI

# 2. Install this node
cd custom_nodes
git clone https://github.com/majidfida/SAM3_SmartInpainter.git

# 3. Install requirements using ComfyUI's embedded Python
..\python_embeded\python.exe -m pip install -r SAM3_SmartInpainter\requirements.txt

# 4. Install SAM3 wrapper (if not already installed)
git clone https://github.com/yolain/ComfyUI-Easy-Sam3.git
```

### For ComfyUI with Virtual Environment (Linux / Mac / Windows venv)

```bash
# 1. Activate your venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Go to custom_nodes and install this node
cd custom_nodes
git clone https://github.com/majidfida/SAM3_SmartInpainter.git

# 3. Install requirements
pip install -r SAM3_SmartInpainter/requirements.txt

# 4. Install SAM3 wrapper (if not already installed)
git clone https://github.com/yolain/ComfyUI-Easy-Sam3.git
```

### requirements.txt Content
```txt
opencv-python>=4.5.0
omegaconf>=2.1.0
hydra-core>=1.1.0
```

---

## 🗂️ Where to Put Your SAM3 Model

Download `sam3.pt` from:  
🔗 https://huggingface.co/Majidfida/Sam3/tree/main

Then place it in:
```
ComfyUI/models/sam3/sam3.pt
```

---

## 🔧 How to Use (Simple Steps)

### Basic Face Inpaint Workflow:

```
[Load Image] → [SAM Smart Inpainter] → [Save Image]
                   ↑
         Connect: MODEL, CLIP, VAE, POSITIVE, NEGATIVE
```

### Important Parameters:

| Parameter | Recommended Value | What It Does |
|-----------|------------------|--------------|
| `prompt` | `"face"` or `"eyes"` | What to find and repaint |
| `text_threshold` | `0.25` – `0.35` | Lower = finds smaller/blurry things |
| `min_mask_area` | `0.001` (0.1%) | Allow tiny masks (for far-away faces) |
| `crop_padding` | `64` – `128` | Give extra space around the mask |
| `mask_dilation` | `6` – `10` | Make thin masks thicker (prevents holes) |
| `inpaint_resolution` | `768` or higher | Better detail in the repainted area |
| `denoise` | `0.4` – `0.6` | How much to change the original (lower = more subtle) |
| `blend_mode` | `"feather"` | Smooth edges when pasting back |

### Tips for Small or Distant Faces:
- Set `min_mask_area` → `0.0005`
- Set `text_threshold` → `0.25`
- Set `crop_padding` → `128`
- Keep `inpaint_resolution` at `768` or higher

### 💡 Tip: Inpaint Multiple Faces (Man + Woman)
Want to repaint **two different faces** in one image? Just **chain multiple nodes**:

```
[Load Image]
       ↓
[SAM Smart Inpainter #1] → prompt: "woman face"
       ↓ (output_image)
[SAM Smart Inpainter #2] → prompt: "man face"
       ↓
[Save Image]
```

Each node finds and repaints only its target. Connect the `output_image` of the first node to the `image` input of the second. Use specific prompts like `"woman face"`, `"man beard"`, `"girl eyes"` to avoid overlap.

---

## ⚙️ Advanced: Use With Any Workflow

This node works like a "smart crop + inpaint" block. You can plug it into:

- Text-to-image pipelines
- Image-to-image upgrades
- Face restoration chains
- Object removal/editing flows

Just connect:
- `MODEL`, `CLIP`, `VAE` from your checkpoint loader
- `positive` / `negative` from your prompt encoders
- `image` from your input

---

## ❗ Important Notes

🔹 **I will not monitor this repository.**  
I may not reply to issues, messages, or pull requests. This node is shared as-is for the community to use and improve.

🔹 **SAM3 only.**  
This node has only been tested with `sam3.pt`. SAM2 support is not included.

🔹 **For face inpainting mainly.**  
This node is designed mainly for faces, but you can type any prompt: `"hat"`, `"logo"`, `"tree"` — it will try to find and repaint it.

🔹 **Got errors?**  
- Make sure you installed **ComfyUI-Easy-Sam3** or **ComfyUI-RMBG**  
- Make sure `sam3.pt` is in `ComfyUI/models/sam3/`  
- Make sure you ran `pip install -r requirements.txt`  
- Check the console log — the node prints helpful debug info

---

## 🙏 Credits

- SAM3 model: Meta AI Research / Facebook (HuggingFace)
- ComfyUI-Easy-Sam3: [@yolain](https://github.com/yolain)
- ComfyUI-RMBG: [@1038lab](https://github.com/1038lab)
- This node: Community contribution

---

> 💡 **Simple rule**: If the mask is too small → lower `min_mask_area`.  
> If the repaint looks blurry → raise `inpaint_resolution`.  
> If edges look harsh → increase `mask_blur` or use `blend_mode: feather`.

Happy inpainting! 🎨✨
