from pathlib import Path

ZDIM = 512
EPOCHS = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_PATH = Path("models")
MODEL_NAME = "ae.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME