import math
from pathlib import Path

import cv2


def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video_cap.get(cv2.CAP_PROP_FPS))

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Cannot read the input video's dimensions.")

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot read the input video's frame rate.")

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create the output video: {output_path}"
        )

    return writer
