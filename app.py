from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
from torchvision import transforms
import timm


IMG_SIZE = 384
MODEL_PATH = Path("model/model_image_spoofing")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "fake_mannequin",
    "fake_mask",
    "fake_printed",
    "fake_screen",
    "fake_unknown",
    "realperson",
]

REAL_CLASS = "realperson"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, dict):
                return value
            if isinstance(value, torch.nn.Module):
                return value

    return checkpoint


def clean_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "model.", "net."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def infer_num_classes(state_dict):
    if not isinstance(state_dict, dict):
        return len(CLASS_NAMES)

    for key in ["head.weight", "head.fc.weight", "classifier.weight", "fc.weight"]:
        if key in state_dict and hasattr(state_dict[key], "shape"):
            if len(state_dict[key].shape) == 2:
                return int(state_dict[key].shape[0])

    for key, value in state_dict.items():
        if key.endswith("weight") and hasattr(value, "shape"):
            if len(value.shape) == 2 and value.shape[0] <= 20:
                return int(value.shape[0])

    return len(CLASS_NAMES)


def build_model(num_classes):
    return timm.create_model(
        "swin_base_patch4_window12_384",
        pretrained=False,
        num_classes=num_classes,
    )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure model/model_image_spoofing exists."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_or_model = extract_state_dict(checkpoint)

    if isinstance(state_or_model, torch.nn.Module):
        model = state_or_model
        model.to(DEVICE)
        model.eval()
        return model

    state_dict = clean_state_dict_keys(state_or_model)
    num_classes = infer_num_classes(state_dict)

    model = build_model(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Detected num_classes: {num_classes}")

    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    return model


model = load_model()


preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def prepare_image(image):
    if image is None:
        raise gr.Error("Upload gambar terlebih dahulu.")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def make_tta_images(image):
    return [
        image,
        ImageOps.mirror(image),
        ImageEnhance.Brightness(image).enhance(1.08),
        ImageEnhance.Contrast(image).enhance(1.08),
    ]


@torch.inference_mode()
def predict(image, use_tta=False):
    image = prepare_image(image)
    images = make_tta_images(image) if use_tta else [image]

    tensors = [preprocess(img) for img in images]
    batch = torch.stack(tensors).to(DEVICE)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    avg_probs = probs.mean(dim=0).detach().cpu()

    num_outputs = len(avg_probs)

    if num_outputs == len(CLASS_NAMES):
        class_names = CLASS_NAMES
    else:
        class_names = [f"class_{i}" for i in range(num_outputs)]

    best_idx = int(torch.argmax(avg_probs).item())
    best_label = class_names[best_idx]
    confidence = float(avg_probs[best_idx])

    if best_label == REAL_CLASS:
        status = "Real Face"
    else:
        status = "Spoof Detected"

    status_text = (
        f"## {status}\n\n"
        f"**Predicted class:** `{best_label}`\n\n"
        f"**Confidence:** `{confidence:.2%}`"
    )

    sorted_probs = sorted(
        [(class_names[i], float(avg_probs[i])) for i in range(num_outputs)],
        key=lambda item: item[1],
        reverse=True,
    )

    prob_lines = [
        f"- `{label}`: **{score:.2%}**"
        for label, score in sorted_probs
    ]

    prob_text = "## Class Probabilities\n\n" + "\n".join(prob_lines)

    return status_text, prob_text


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload face image"),
        gr.Checkbox(label="Use TTA / Test-Time Augmentation", value=False),
    ],
    outputs=[
        gr.Markdown(label="Result"),
        gr.Markdown(label="Class probabilities"),
    ],
    title="Image Spoofing Detection",
    description=(
        "Upload a face image to classify whether it is a real face or a spoofing attempt. "
        "The model predicts spoof categories such as printed image, screen replay, mask, "
        "mannequin, unknown spoof, or real person."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()