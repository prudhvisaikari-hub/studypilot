import os
import wave
import struct
import math
from PIL import Image, ImageDraw, ImageFont

scratch_dir = os.path.join(os.path.dirname(__file__), "test_assets")
os.makedirs(scratch_dir, exist_ok=True)

# 1. Generate 5-second audio WAV file
sample_rate = 16000
duration = 5.0
n_samples = int(sample_rate * duration)
wav_path = os.path.join(scratch_dir, "sample_audio.wav")

with wave.open(wav_path, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(n_samples):
        # 440 Hz tone
        value = int(10000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        f.writeframes(struct.pack("<h", value))

# 2. Generate 0.5-second short audio WAV file
short_wav_path = os.path.join(scratch_dir, "short_audio.wav")
with wave.open(short_wav_path, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(int(sample_rate * 0.5)):
        value = int(5000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        f.writeframes(struct.pack("<h", value))

# 3. Generate silent audio WAV file
silent_wav_path = os.path.join(scratch_dir, "silent_audio.wav")
with wave.open(silent_wav_path, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for i in range(sample_rate * 2):
        f.writeframes(struct.pack("<h", 0))

# 4. Generate 0-byte empty audio file
empty_wav_path = os.path.join(scratch_dir, "empty_audio.wav")
with open(empty_wav_path, "wb") as f:
    pass

# 5. Generate sample slide image with Pillow
slide_path = os.path.join(scratch_dir, "sample_slide.png")
img = Image.new("RGB", (800, 600), color=(255, 255, 255))
draw = ImageDraw.Draw(img)
draw.rectangle([(20, 20), (780, 580)], outline=(0, 0, 0), width=3)
draw.text((50, 50), "Physics 101: Newton's Laws", fill=(0, 0, 0))
draw.text((50, 120), "1. First Law: Law of Inertia", fill=(0, 0, 0))
draw.text((50, 180), "2. Second Law: F = m * a", fill=(0, 0, 0))
draw.text((50, 240), "3. Third Law: Action = Reaction", fill=(0, 0, 0))
draw.text((50, 320), "Key Terms: Force, Mass, Acceleration, Momentum", fill=(0, 0, 0))
img.save(slide_path)

print("Test assets created successfully in:", scratch_dir)
