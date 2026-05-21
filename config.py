import torch

IMG_SIZE = 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL_PATH = "emotion_model.pth"
EMOTION_LABELS = {
    0: "生气",
    1: "厌恶",
    2: "恐惧",
    3: "开心",
    4: "难过",
    5: "惊讶",
    6: "中性"
}