"""
SAM Smart Inpainter — ComfyUI Custom Node  v0.3 (Fixed latent_image API + decode)
All‑in‑one text‑prompted segmentation + diffusion inpainting node.
Supports SAM3  models.

"""

import os
import sys
import math
import traceback
import torch
import numpy as np
import safetensors.torch
import folder_paths
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.model_management
from PIL import Image

# ============================================================================
# 1.  Model folder registration
# ============================================================================
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))


def _ensure_model_folder(key, subfolder):
    try:
        base = folder_paths.models_dir
    except AttributeError:
        base = os.path.join(os.path.dirname(os.path.dirname(_NODE_DIR)), "models")
    path = os.path.join(base, subfolder)
    os.makedirs(path, exist_ok=True)
    exts = getattr(folder_paths, "supported_pt_extensions",
                   {".pt", ".pth", ".bin", ".safetensors", ".ckpt"})
    if key not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[key] = ([path], exts)
    elif path not in folder_paths.folder_names_and_paths[key][0]:
        folder_paths.folder_names_and_paths[key][0].append(path)


_ensure_model_folder("sam3", "sam3")
_ensure_model_folder("sam2", "sam2")
_ensure_model_folder("sams", "sams")
_SAM_SEARCH_KEYS = ["sam3", "sams", "sam2", "checkpoints"]


def _list_all_sam_models():
    seen, out = set(), []
    for key in _SAM_SEARCH_KEYS:
        try:
            for name in folder_paths.get_filename_list(key):
                if name not in seen:
                    seen.add(name)
                    out.append(name)
        except Exception:
            pass
    return out or ["<no models found>"]


def _resolve_path(filename):
    for key in _SAM_SEARCH_KEYS:
        try:
            p = folder_paths.get_full_path(key, filename)
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
    return None


# ============================================================================
# 2.  Model‑family detection
# ============================================================================
def _is_sam3_model(path):
    name = os.path.basename(path).lower()
    return "sam3" in name or "multiplex" in name


# ============================================================================
# 3.  Safe weight loading
# ============================================================================
def _load_state_dict_safe(path):
    try:
        return safetensors.torch.load_file(path)
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n{type(e).__name__}: {e}"
        )


# ============================================================================
# 4.  SAM3 package discovery
# ============================================================================
def _find_sam3_package():
    try:
        import sam3
        return None
    except ImportError:
        pass
    comfy_root = os.path.dirname(os.path.dirname(_NODE_DIR))
    nodes_root = os.path.join(comfy_root, "custom_nodes")
    search_roots = [nodes_root]
    custom = folder_paths.folder_names_and_paths.get("custom_nodes")
    if custom:
        search_roots.extend(custom[0])

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pack in os.listdir(root):
            sam3_dir = os.path.join(root, pack, "sam3")
            init_py = os.path.join(sam3_dir, "__init__.py")
            if os.path.isdir(sam3_dir) and os.path.isfile(init_py):
                parent = os.path.join(root, pack)
                if parent not in sys.path:
                    sys.path.insert(0, parent)
                try:
                    import sam3
                    return parent
                except ImportError:
                    if parent in sys.path:
                        sys.path.remove(parent)
    return None


_sam3_pkg_added = _find_sam3_package()
try:
    from sam3.model_builder import build_sam3_image_model as _build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
    _SAM3_AVAILABLE = True
except ImportError:
    _SAM3_AVAILABLE = False
    _build_sam3_image_model = None
    _Sam3Processor = None


# ============================================================================
# 5.  BPE tokenizer locator
# ============================================================================
def _find_sam3_bpe_path():
    target = "bpe_simple_vocab_16e6.txt.gz"
    try:
        import sam3 as _s
        pkg_dir = os.path.dirname(_s.__file__)
        candidates = [
            os.path.join(pkg_dir, "assets", target),
            os.path.join(pkg_dir, "..", "assets", target),
        ]
        for c in candidates:
            c = os.path.normpath(c)
            if os.path.isfile(c):
                return c
    except Exception:
        pass
    comfy_root = os.path.dirname(os.path.dirname(_NODE_DIR))
    nodes_root = os.path.join(comfy_root, "custom_nodes")
    if os.path.isdir(nodes_root):
        for pack in os.listdir(nodes_root):
            for sub in (
                os.path.join(nodes_root, pack, "sam3", "assets", target),
                os.path.join(nodes_root, pack, "models", "sam3", "assets", target),
                os.path.join(nodes_root, pack, "assets", target),
            ):
                if os.path.isfile(sub):
                    return sub
    return None


# ============================================================================
# 6.  SAM3 loader
# ============================================================================
def _load_sam3_predictor(path, device):
    if not _SAM3_AVAILABLE:
        raise RuntimeError(
            "sam3 package not found. Install ComfyUI‑Easy‑Sam3 or ComfyUI‑RMBG."
        )
    bpe_path = _find_sam3_bpe_path()
    device_str = str(device)

    err_a = "not attempted"
    try:
        kwargs = dict(eval_mode=True, checkpoint_path=path, load_from_HF=False)
        if bpe_path:
            kwargs["bpe_path"] = bpe_path
        try:
            model = _build_sam3_image_model(device=device_str, **kwargs)
        except TypeError:
            model = _build_sam3_image_model(**kwargs)
        model = model.to(device).float()
        model.eval()
        processor = _make_sam3_processor(model, device_str)
        return _SAM3Wrapper(processor, device)
    except Exception as e:
        err_a = f"{type(e).__name__}: {e}"

    err_b = "not attempted"
    try:
        build_kwargs = dict(load_from_HF=False, eval_mode=True)
        if bpe_path:
            build_kwargs["bpe_path"] = bpe_path
        model = _build_sam3_image_model(**build_kwargs)
        sd = _load_state_dict_safe(path)

        def _remap(state_dict):
            out = {}
            for k, v in state_dict.items():
                if k.startswith("detector."):
                    out[k[len("detector."):]] = v
                elif not k.startswith("tracker."):
                    out[k] = v
            return out

        loaded = False
        for sd_try in [sd, _remap(sd)]:
            try:
                model.load_state_dict(sd_try, strict=False)
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            raise RuntimeError("load_state_dict failed for all remapping strategies")

        model = model.to(device).float()
        model.eval()
        processor = _make_sam3_processor(model, device_str)
        return _SAM3Wrapper(processor, device)
    except Exception as e:
        err_b = f"{type(e).__name__}: {e}"

    err_c = "not attempted"
    try:
        model = _build_sam3_image_model()
        model = model.to(device).float()
        model.eval()
        processor = _make_sam3_processor(model, device_str)
        return _SAM3Wrapper(processor, device)
    except Exception as e:
        err_c = f"{type(e).__name__}: {e}"

    raise RuntimeError(
        f"All three SAM3 load strategies failed for: {os.path.basename(path)}\n"
        f"  A (explicit params): {err_a}\n"
        f"  B (manual sd load):  {err_b}\n"
        f"  C (default / HF):    {err_c}\n"
        f"BPE file found: {bpe_path or 'NOT FOUND'}"
    )


def _make_sam3_processor(model, device_str):
    try:
        return _Sam3Processor(model, device=device_str)
    except TypeError:
        return _Sam3Processor(model)


# ============================================================================
# 7.  SAM2 loader
# ============================================================================
def _sam2_yaml_name(model_path):
    name = os.path.basename(model_path).lower()
    v21 = "2.1" in name or "sam2.1" in name
    if "large" in name or "_l." in name or "hiera_l" in name:
        sz = "l"
    elif "base_p" in name or "_b+" in name or "hiera_b" in name:
        sz = "b+"
    elif "small" in name or "_s." in name or "hiera_s" in name:
        sz = "s"
    elif "tiny" in name or "_t." in name or "hiera_t" in name:
        sz = "t"
    else:
        sz = "l"
    return ("sam2.1", f"sam2.1_hiera_{sz}.yaml") if v21 else ("sam2", f"sam2_hiera_{sz}.yaml")


def _find_sam2_yaml(model_path):
    sub, fname = _sam2_yaml_name(model_path)
    candidates = []
    try:
        import sam2 as _s2
        d = os.path.dirname(_s2.__file__)
        candidates += [
            os.path.join(d, "configs", sub, fname),
            os.path.join(d, "configs", fname),
            os.path.join(d, fname),
        ]
    except ImportError:
        pass
    comfy_root = os.path.dirname(os.path.dirname(_NODE_DIR))
    nodes_root = os.path.join(comfy_root, "custom_nodes")
    if os.path.isdir(nodes_root):
        for pack in os.listdir(nodes_root):
            for cfgdir in ("sam2_configs", "configs"):
                candidates += [
                    os.path.join(nodes_root, pack, cfgdir, fname),
                    os.path.join(nodes_root, pack, cfgdir, sub, fname),
                ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _load_sam2_predictor(path, device):
    yaml_path = _find_sam2_yaml(path)
    if yaml_path is None:
        return _load_sam2_via_build_sam2(path, device)
    try:
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        cfg = OmegaConf.load(yaml_path)
        model = instantiate(cfg.model, _recursive_=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate SAM2 model from YAML '{yaml_path}': {e}"
        )

    sd = _load_state_dict_safe(path)

    def _try_load(state_dict, strict):
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            return not (strict and (missing or unexpected))
        except Exception:
            return False

    loaded = (
        _try_load(sd, strict=True)
        or _try_load({k.lstrip("model."): v for k, v in sd.items()}, strict=False)
        or _try_load(sd, strict=False)
    )
    if not loaded:
        raise RuntimeError("SAM2 state_dict load failed.")

    model.to(device=device, dtype=torch.float32)
    model.eval()

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(model)
    except ImportError:
        raise RuntimeError("sam2 package not found. Install: pip install sam2")
    return _SAM2Wrapper(predictor, device)


def _load_sam2_via_build_sam2(path, device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise RuntimeError("sam2 package not found.")
    sub, fname = _sam2_yaml_name(path)
    fname_noext = fname.replace(".yaml", "")
    candidates = [f"configs/{sub}/{fname}", f"configs/{fname}", fname, fname_noext]
    err_last = None
    for cfg_str in candidates:
        try:
            sam = build_sam2(cfg_str, path, device=device)
            return _SAM2Wrapper(SAM2ImagePredictor(sam), device)
        except Exception as e:
            err_last = e
    raise RuntimeError(f"build_sam2 failed. Last error: {err_last}")


# ============================================================================
# 8.  Unified predictor builder
# ============================================================================
def _build_predictor(path, device):
    return (
        _load_sam3_predictor(path, device)
        if _is_sam3_model(path)
        else _load_sam2_predictor(path, device)
    )


# ============================================================================
# 9.  Predictor wrappers
# ============================================================================
class _SAM3Wrapper:
    family = "sam3"

    def __init__(self, processor, device):
        self.processor = processor
        self.device = device
        self._state = None

    def set_image(self, img_np_rgb):
        pil = Image.fromarray(img_np_rgb.astype(np.uint8))
        self._state = self.processor.set_image(pil)

    def predict_with_text(self, prompt, threshold=0.35):
        if self._state is None:
            return []
        try:
            output = self.processor.set_text_prompt(state=self._state, prompt=prompt)
            masks = output.get("masks", [])
            scores = output.get("scores", [])
            result = []
            for i, m in enumerate(masks):
                m = m.cpu().numpy() if isinstance(m, torch.Tensor) else np.asarray(m, dtype=bool)
                if m.ndim == 3:
                    m = m[0]
                score = float(scores[i]) if i < len(scores) else 1.0
                if score >= threshold:
                    result.append((m, score))
            return result
        except Exception as e:
            print(f"[SAM_SmartInpainter] SAM3 text predict error: {e}")
            return []

    def predict_with_point(self, point_xy):
        if self._state is None:
            return []
        try:
            output = self.processor.set_point_prompt(
                state=self._state, points=[list(point_xy)], labels=[1]
            )
            masks = output.get("masks", [])
            scores = output.get("scores", [])
            return [
                (
                    m.cpu().numpy()[0] if isinstance(m, torch.Tensor) else np.asarray(m, dtype=bool),
                    float(scores[i]) if i < len(scores) else 1.0,
                )
                for i, m in enumerate(masks)
            ]
        except Exception:
            return []


class _SAM2Wrapper:
    family = "sam2"

    def __init__(self, predictor, device):
        self.predictor = predictor
        self.device = device

    def set_image(self, img_np_rgb):
        if hasattr(self.predictor, "set_image"):
            self.predictor.set_image(img_np_rgb)
        elif hasattr(self.predictor, "set_torch_image"):
            t = torch.from_numpy(img_np_rgb).permute(2, 0, 1).float() / 255.0
            self.predictor.set_torch_image(t, img_np_rgb.shape[:2])

    def predict_with_boxes(self, boxes_px):
        results = []
        for box in boxes_px:
            box_arr = np.array(box, dtype=np.float32)
            try:
                masks, scores, _ = self.predictor.predict(
                    box=box_arr, multimask_output=False, return_logits=False
                )
            except TypeError:
                masks, scores, _ = self.predictor.predict(
                    box=box_arr, multimask_output=False
                )
            if len(masks):
                results.append((masks[0].astype(bool), float(scores[0])))
        return results

    def predict_with_point(self, point_xy):
        pt = np.array([list(point_xy)], dtype=np.float32)
        lbl = np.array([1], dtype=np.int32)
        try:
            masks, scores, _ = self.predictor.predict(
                point_coords=pt, point_labels=lbl,
                multimask_output=False, return_logits=False
            )
        except TypeError:
            masks, scores, _ = self.predictor.predict(
                point_coords=pt, point_labels=lbl, multimask_output=False
            )
        return [(masks[0].astype(bool), float(scores[0]))] if len(masks) else []


# ============================================================================
# 10. Optional Grounding‑DINO
# ============================================================================
_GDINO_AVAILABLE = False
try:
    from groundingdino.util.inference import load_model as _gdino_load
    from groundingdino.util.inference import predict as _gdino_predict
    _GDINO_AVAILABLE = True
except Exception:
    pass


def _gdino_boxes(image_np_rgb, text, threshold=0.30, device="cuda"):
    if not _GDINO_AVAILABLE:
        return []
    try:
        from groundingdino.util.inference import load_image
        pil = Image.fromarray(image_np_rgb)
        img_t, _ = load_image(pil)
        ckpt = _find_gdino_ckpt()
        if ckpt is None:
            return []
        model = _gdino_load(
            "groundingdino/config/GroundingDINO_SwinT_OGC.py", ckpt
        ).to(device)
        boxes, logits, _ = _gdino_predict(model, img_t, text, threshold, threshold)
        h, w = image_np_rgb.shape[:2]
        out = []
        for b in boxes:
            cx, cy, bw, bh = b.tolist()
            out.append([
                max(0, int((cx - bw / 2) * w)),
                max(0, int((cy - bh / 2) * h)),
                min(w, int((cx + bw / 2) * w)),
                min(h, int((cy + bh / 2) * h)),
            ])
        return out
    except Exception as e:
        print(f"[SAM_SmartInpainter] GDino error: {e}")
        return []


def _find_gdino_ckpt():
    for key in ("checkpoints", "models"):
        try:
            folders = folder_paths.get_folder_paths(key)
        except Exception:
            continue
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if "groundingdino" in f.lower() and f.endswith((".pth", ".pt", ".bin")):
                    return os.path.join(folder, f)
    return None


# ============================================================================
# 11. Image / mask utilities & Conditioning Normalizer
# ============================================================================
def _gaussian_blur_mask(mask_t, blur_px):
    if blur_px <= 0:
        return mask_t
    k = blur_px * 2 + 1
    try:
        import torchvision.transforms.functional as TF
        return TF.gaussian_blur(
            mask_t.unsqueeze(0).unsqueeze(0), kernel_size=k, sigma=float(blur_px)
        ).squeeze()
    except Exception:
        return mask_t


def _erode_dilate(mask_np, erosion, dilation):
    try:
        import cv2
        m = (mask_np * 255).astype(np.uint8)
        if erosion > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion * 2 + 1,) * 2)
            m = cv2.erode(m, k)
        if dilation > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1,) * 2)
            m = cv2.dilate(m, k)
        return m.astype(np.float32) / 255.0
    except Exception:
        return mask_np


def _snap8(v):
    return math.ceil(v / 8) * 8


def _normalize_conditioning(cond):
    """
    Ensure conditioning is in ComfyUI's canonical format: list of (tensor, dict) tuples.
    Handles all known formats produced by CLIP encode nodes across ComfyUI versions.
    """
    if isinstance(cond, list) and len(cond) > 0:
        first = cond[0]
        # Already canonical: list of (tensor, dict) tuples
        if isinstance(first, (list, tuple)) and len(first) == 2:
            tensor_part, extra_part = first
            if isinstance(tensor_part, torch.Tensor) and isinstance(extra_part, dict):
                return cond
        # Old dict-list format
        result = []
        for item in cond:
            if isinstance(item, dict):
                tensor = item.get("cond") or item.get("samples")
                extra = {}
                if "pooled_output" in item:
                    extra["pooled_output"] = item["pooled_output"]
                result.append((tensor, extra))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result.append(tuple(item))
            else:
                result.append((item, {}))
        return result
    elif isinstance(cond, dict):
        tensor = cond.get("cond") or cond.get("samples")
        extra = {}
        if "pooled_output" in cond:
            extra["pooled_output"] = cond["pooled_output"]
        return [(tensor, extra)]
    elif isinstance(cond, torch.Tensor):
        return [(cond, {})]
    return cond


def _prepare_noise_mask_for_latent(mask_pixel, latent_tensor, feather_px):
    """
    Take a pixel-space mask and produce a latent-space noise_mask tensor
    with shape [1, H_latent, W_latent] ready for comfy.sample.sample.
    """
    # mask_pixel: [H_pixel, W_pixel]
    nm = mask_pixel.unsqueeze(0)  # [1, H, W]
    if feather_px > 0:
        nm = _gaussian_blur_mask(nm.squeeze(0), feather_px).unsqueeze(0)

    lh, lw = latent_tensor.shape[2], latent_tensor.shape[3]
    # Interpolate to latent spatial size: [1, 1, lh, lw] → squeeze → [1, lh, lw]
    noise_mask = torch.nn.functional.interpolate(
        nm.unsqueeze(0).float(),
        size=(lh, lw),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)  # [1, lh, lw]
    return noise_mask.clamp(0.0, 1.0)


# ============================================================================
# 12. Main node: SAM Smart Inpainter
# ============================================================================
class SAM_SmartInpainter:
    @classmethod
    def INPUT_TYPES(cls):
        models = _list_all_sam_models()
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (
                    models,
                    {"tooltip": "SAM3 → models/sam3/ | SAM2 → models/sam2/ or models/sams/"},
                ),
                "prompt": ("STRING", {"default": "face", "multiline": True}),
            },
            "optional": {
                "text_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "fallback_point_mode": (["center", "face_region", "custom"], {"default": "face_region"}),
                "custom_point_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "custom_point_y": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "multi_mask_merge": (["union", "largest", "highest_score"], {"default": "union"}),
                "min_mask_area": ("FLOAT", {"default": 0.005, "min": 0.0001, "max": 0.5, "step": 0.001}),
                "max_mask_area": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 64}),
                "mask_erosion": ("INT", {"default": 0, "min": 0, "max": 32}),
                "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 64}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "crop_padding": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8}),
                "crop_padding_mode": (["aspect", "square"], {"default": "aspect"}),
                "inpaint_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "upscale_method": (["bicubic", "bilinear", "lanczos", "nearest"], {"default": "bicubic"}),
                "blend_mode": (["feather", "mask", "hard"], {"default": "feather"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "karras"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "denoise": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "noise_mask_feather": ("INT", {"default": 8, "min": 0, "max": 64}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("output_image", "inpainted_patch", "segmentation_mask", "debug_info")
    FUNCTION = "execute"
    CATEGORY = "image/inpainting"

    # ------------------------------------------------------------------
    def execute(self, image, sam_model, prompt, **kwargs):
        log = []
        device = (
            comfy.model_management.get_torch_device()
            if kwargs.get("device", "auto") == "auto"
            else torch.device(kwargs["device"])
        )
        log.append(f"Device: {device}")

        img_t = image[0]
        img_np = (img_t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        H, W = img_np.shape[:2]

        model_path = _resolve_path(sam_model)
        if model_path is None:
            return self._passthrough(image, f"Model file not found: {sam_model}")

        family = "sam3" if _is_sam3_model(model_path) else "sam2"
        log.append(f"Family: {family.upper()}")

        try:
            predictor = _build_predictor(model_path, device)
            predictor.set_image(img_np)
        except Exception as e:
            return self._passthrough(
                image, f"Failed to load model: {e}\n{traceback.format_exc()[:500]}"
            )
        log.append("Model loaded OK")

        masks_and_scores = self._get_masks(
            predictor, img_np, prompt.strip(), H, W,
            kwargs.get("text_threshold", 0.35),
            kwargs.get("fallback_point_mode", "face_region"),
            kwargs.get("custom_point_x", 0.5),
            kwargs.get("custom_point_y", 0.25),
            device, log,
        )

        if not masks_and_scores:
            return self._passthrough(image, f"No masks produced for: '{prompt}'")

        masks_only = [m for m, _ in masks_and_scores]
        scores = [s for _, s in masks_and_scores]
        merge = kwargs.get("multi_mask_merge", "union")
        if merge == "largest":
            idx = max(range(len(masks_only)), key=lambda i: masks_only[i].sum())
            merged = masks_only[idx].astype(np.float32)
        elif merge == "highest_score":
            idx = max(range(len(scores)), key=lambda i: scores[i])
            merged = masks_only[idx].astype(np.float32)
        else:
            merged = np.zeros((H, W), dtype=np.float32)
            for m in masks_only:
                merged = np.maximum(merged, m.astype(np.float32))

        log.append(f"Raw mask coverage: {merged.mean() * 100:.1f}%")
        merged = _erode_dilate(
            merged, kwargs.get("mask_erosion", 0), kwargs.get("mask_dilation", 4)
        )
        if kwargs.get("invert_mask", False):
            merged = 1.0 - merged

        mask_t = torch.from_numpy(merged).float()
        mask_t = _gaussian_blur_mask(mask_t, kwargs.get("mask_blur", 6))
        mask_bin = (mask_t > kwargs.get("mask_threshold", 0.5)).float()
        frac = mask_bin.mean().item()
        log.append(f"Final mask: {frac * 100:.1f}%")

        min_area = kwargs.get("min_mask_area", 0.005)
        max_area = kwargs.get("max_mask_area", 0.95)
        if frac < min_area:
            return self._passthrough(
                image, f"Mask too small ({frac * 100:.2f}% < {min_area * 100:.2f}%)"
            )
        if frac > max_area:
            return self._passthrough(
                image, f"Mask too large ({frac * 100:.2f}% > {max_area * 100:.2f}%)"
            )

        y_idx, x_idx = torch.where(mask_bin > 0.5)
        if len(x_idx) == 0:
            return self._passthrough(image, "No valid mask pixels")

        # ------------------------------------------------------------------
        # Crop region around the mask
        # ------------------------------------------------------------------
        pad = kwargs.get("crop_padding", 32)
        x1r = max(0, x_idx.min().item() - pad)
        x2r = min(W, x_idx.max().item() + pad)
        y1r = max(0, y_idx.min().item() - pad)
        y2r = min(H, y_idx.max().item() + pad)

        if kwargs.get("crop_padding_mode", "aspect") == "square":
            side = max(x2r - x1r, y2r - y1r)
            cx, cy = (x1r + x2r) // 2, (y1r + y2r) // 2
            x1 = max(0, cx - side // 2)
            x2 = min(W, cx + side // 2)
            y1 = max(0, cy - side // 2)
            y2 = min(H, cy + side // 2)
        else:
            x1, x2, y1, y2 = x1r, x2r, y1r, y2r

        x2 = min(W, x1 + _snap8(x2 - x1))
        y2 = min(H, y1 + _snap8(y2 - y1))
        log.append(f"Crop: ({x1},{y1})→({x2},{y2})")

        crop_img = image[:, y1:y2, x1:x2, :]   # [1, ch, cw, 3]
        crop_mask = mask_t[y1:y2, x1:x2]        # [ch, cw]
        crop_h, crop_w = y2 - y1, x2 - x1

        # ------------------------------------------------------------------
        # Upscale crop to inpaint resolution
        # ------------------------------------------------------------------
        iw_base = _snap8(crop_w)
        ih_base = _snap8(crop_h)
        target = kwargs.get("inpaint_resolution", 768)
        scale = target / max(iw_base, ih_base)
        iw = _snap8(int(iw_base * scale))
        ih = _snap8(int(ih_base * scale))

        up_method = kwargs.get("upscale_method", "bicubic")
        resized_img = comfy.utils.common_upscale(
            crop_img.movedim(-1, 1), iw, ih, up_method, "disabled"
        ).movedim(1, -1)  # [1, ih, iw, 3]
        resized_mask = comfy.utils.common_upscale(
            crop_mask.unsqueeze(0).unsqueeze(0), iw, ih, up_method, "disabled"
        ).squeeze(0).squeeze(0)  # [ih, iw]

        # Default result = the (possibly upscaled) crop (no inpainting)
        result_patch = resized_img

        # ------------------------------------------------------------------
        # Diffusion inpainting
        # ------------------------------------------------------------------
        diff_model = kwargs.get("model")
        clip = kwargs.get("clip")
        vae = kwargs.get("vae")
        positive = kwargs.get("positive")
        negative = kwargs.get("negative")

        if all(x is not None for x in [diff_model, vae, clip, positive, negative]):
            try:
                # Step 1 — normalise conditioning
                positive = _normalize_conditioning(positive)
                negative = _normalize_conditioning(negative)
                log.append("Conditioning normalised OK")

                # Step 2 — VAE encode the cropped+upscaled image
                encode_input = resized_img[..., :3]  # drop alpha if present
                latent_img = vae.encode(encode_input)  # returns Tensor [B, C, H, W]
                log.append(f"VAE encode OK — latent shape: {list(latent_img.shape)}")

                # Step 3 — optional fix_empty_latent_channels (newer ComfyUI)
                if hasattr(comfy.sample, "fix_empty_latent_channels"):
                    latent_img = comfy.sample.fix_empty_latent_channels(diff_model, latent_img)

                # Step 4 — build noise mask in latent space  [1, H_lat, W_lat]
                noise_mask = _prepare_noise_mask_for_latent(
                    resized_mask, latent_img, kwargs.get("noise_mask_feather", 8)
                )
                log.append(f"Noise mask shape: {list(noise_mask.shape)}")

                # Step 5 — prepare noise
                seed = kwargs.get("seed", 0)
                actual_seed = (
                    seed if seed != 0 else torch.randint(0, 2**32, (1,)).item()
                )
                noise = comfy.sample.prepare_noise(latent_img, actual_seed)
                log.append(f"Noise prepared (seed {actual_seed})")

                # ----------------------------------------------------------
                # Step 6 — run the sampler
                #
                # CRITICAL FIX (v5.3):
                #   comfy.sample.sample() expects:
                #     latent_image  →  a raw TENSOR  (NOT {"samples": tensor})
                #   and returns:
                #     samples       →  a raw TENSOR  (NOT a dict)
                #
                #   Passing the dict caused:
                #     'dict' object has no attribute 'is_nested'
                #   because PyTorch / ComfyUI internally calls .is_nested on
                #   what it expects to be a Tensor.
                # ----------------------------------------------------------
                samples = comfy.sample.sample(
                    model=diff_model,
                    noise=noise,
                    steps=kwargs.get("steps", 20),
                    cfg=kwargs.get("cfg", 7.0),
                    sampler_name=kwargs.get("sampler_name", "dpmpp_2m"),
                    scheduler=kwargs.get("scheduler", "karras"),
                    positive=positive,
                    negative=negative,
                    latent_image=latent_img,   # ← RAW TENSOR (not dict)
                    denoise=kwargs.get("denoise", 0.55),
                    disable_noise=False,
                    start_step=None,
                    last_step=None,
                    force_full_denoise=False,
                    noise_mask=noise_mask,      # ← separate kwarg, shape [1,H,W]
                )
                log.append(f"Sampling OK — output shape: {list(samples.shape)}")

                # Step 7 — VAE decode: samples is a Tensor, decode it directly
                result_patch = vae.decode(samples)  # ← NOT samples["samples"]
                log.append(f"VAE decode OK — patch shape: {list(result_patch.shape)}")

            except Exception as e:
                log.append(f"Inpainting failed: {e}")
                log.append(traceback.format_exc()[:600])
        else:
            log.append("Inpainting skipped — diffusion inputs not connected")

        # ------------------------------------------------------------------
        # Downscale result_patch back to original crop size, blend into image
        # ------------------------------------------------------------------
        restored = comfy.utils.common_upscale(
            result_patch.movedim(-1, 1), crop_w, crop_h, up_method, "disabled"
        ).movedim(1, -1)  # [1, crop_h, crop_w, 3]

        output = image.clone()
        blend_strength = kwargs.get("blend_strength", 1.0)
        blend_mode = kwargs.get("blend_mode", "feather")

        if blend_mode == "hard":
            blend_mask = (crop_mask > kwargs.get("mask_threshold", 0.5)).float()
        else:
            blend_mask = crop_mask.clamp(0, 1)

        bm = (blend_mask * blend_strength).unsqueeze(-1).unsqueeze(0)  # [1,H,W,1]
        output[:, y1:y2, x1:x2, :] = (
            output[:, y1:y2, x1:x2, :] * (1.0 - bm) + restored * bm
        )

        mask_out = mask_t.unsqueeze(0)  # [1, H, W]
        return (output, result_patch, mask_out, "\n".join(log))

    # ------------------------------------------------------------------
    def _get_masks(
        self, predictor, img_np, prompt, H, W,
        threshold, fallback_point_mode, cx_norm, cy_norm, device, log
    ):
        if predictor.family == "sam3":
            log.append("Using SAM3 native text prompt")
            results = predictor.predict_with_text(prompt, threshold=threshold)
            if not results:
                log.append("SAM3 text returned no masks — trying point fallback")
                results = predictor.predict_with_point((W // 2, H // 2))
            return results
        else:
            log.append(f"GDino available: {_GDINO_AVAILABLE}")
            boxes = (
                _gdino_boxes(img_np, prompt, threshold, str(device))
                if _GDINO_AVAILABLE
                else []
            )
            log.append(f"GDino detected {len(boxes)} box(es)")
            if boxes:
                return predictor.predict_with_boxes(boxes)
            if fallback_point_mode == "custom":
                px, py = int(cx_norm * W), int(cy_norm * H)
            elif fallback_point_mode == "face_region":
                px, py = W // 2, int(H * 0.25)
            else:
                px, py = W // 2, H // 2
            log.append(f"SAM2 point fallback: ({px},{py})")
            return predictor.predict_with_point((px, py))

    def _passthrough(self, image, msg):
        print(f"[SAM_SmartInpainter] {msg}")
        return (
            image,
            image[:1],
            torch.zeros(1, image.shape[1], image.shape[2]),
            msg,
        )


# ============================================================================
# 13. Mask‑only node
# ============================================================================
class SAM_MaskOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_model": (_list_all_sam_models(), {}),
                "prompt": ("STRING", {"default": "face", "multiline": True}),
            },
            "optional": {
                "text_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "mask_blur": ("INT", {"default": 6, "min": 0, "max": 64}),
                "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 64}),
                "mask_erosion": ("INT", {"default": 0, "min": 0, "max": 32}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "mask_preview", "debug_info")
    FUNCTION = "execute"
    CATEGORY = "image/inpainting"

    def execute(
        self, image, sam_model, prompt,
        text_threshold=0.35, mask_blur=6, mask_dilation=4,
        mask_erosion=0, invert_mask=False, device="auto"
    ):
        log = []
        dev = (
            comfy.model_management.get_torch_device()
            if device == "auto"
            else torch.device(device)
        )
        img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        if img_np.shape[-1] == 4:
            img_np = img_np[..., :3]
        H, W = img_np.shape[:2]

        model_path = _resolve_path(sam_model)
        if model_path is None:
            return (torch.zeros(1, H, W), image, f"Model not found: {sam_model}")
        try:
            predictor = _build_predictor(model_path, dev)
            predictor.set_image(img_np)
        except Exception as e:
            return (torch.zeros(1, H, W), image, f"Model load error: {e}")

        if predictor.family == "sam3":
            results = predictor.predict_with_text(prompt.strip(), threshold=text_threshold)
            if not results:
                results = predictor.predict_with_point((W // 2, H // 2))
        else:
            boxes = (
                _gdino_boxes(img_np, prompt.strip(), text_threshold, str(dev))
                if _GDINO_AVAILABLE
                else []
            )
            results = (
                predictor.predict_with_boxes(boxes)
                if boxes
                else predictor.predict_with_point((W // 2, H // 2))
            )

        if not results:
            return (torch.zeros(1, H, W), image, f"No masks found for: '{prompt}'")

        merged = np.zeros((H, W), dtype=np.float32)
        for m, _ in results:
            merged = np.maximum(merged, m.astype(np.float32))
        if mask_erosion > 0 or mask_dilation > 0:
            merged = _erode_dilate(merged, mask_erosion, mask_dilation)
        if invert_mask:
            merged = 1.0 - merged

        mask_t = torch.from_numpy(merged).float()
        mask_t = _gaussian_blur_mask(mask_t, mask_blur)
        log.append(f"Coverage: {mask_t.mean().item() * 100:.1f}%")

        mask_preview = mask_t.unsqueeze(-1).expand(H, W, 3).unsqueeze(0)
        return (mask_t.unsqueeze(0), mask_preview, "\n".join(log))


# ============================================================================
# 14. Node registration
# ============================================================================
NODE_CLASS_MAPPINGS = {
    "SAM_SmartInpainter": SAM_SmartInpainter,
    "SAM_MaskOnly": SAM_MaskOnly,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM_SmartInpainter": "SAM Smart Inpainter (SAM3 v0.3)",
    "SAM_MaskOnly": "SAM Mask Generator (SAM3 v0.3)",
}
