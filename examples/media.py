"""Classify a local image, or sample four frames from a local video."""
import argparse
from valen_sdk import Valen

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Image or video file")
parser.add_argument("--video", action="store_true")
args = parser.parse_args()
part = {"type": "input_video", "video_url": args.path} if args.video else {
    "type": "input_image", "image_url": args.path}
question = {
    "name": "sauce", "type": "choice",
    "instructions": "What color is the sauce in the foil container on the left?",
    "choices": [{"value": color, "description": color.title()}
                for color in ["brown", "white", "red", "orange"]],
} if args.video else {
    "name": "animal", "type": "choice", "instructions": "Which animals are shown?",
    "choices": [{"value": "sheep", "description": "Sheep"}, {"value": "dogs", "description": "Dogs"}],
}
with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32", media_kwargs={
    "videos_kwargs": {"num_frames": 4, "fps": None},
}) as client:
    result = client.decisions(input=[{"role": "user", "content": [part]}], questions=[question])
    print("Video answer:" if args.video else "Image answer:", result.answers[0].choice)
