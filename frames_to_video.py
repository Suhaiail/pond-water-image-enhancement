import argparse
import os
import re
from glob import glob
from typing import List

import cv2


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_frame_pngs(folder: str) -> List[str]:
    paths = glob(os.path.join(folder, "*.png"))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort(key=natural_key)
    return paths


def resize_to_match(frame, width: int, height: int):
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description="Convert all frame PNG images in a folder into one MP4 file.")
    ap.add_argument("--input", required=True, help="Folder containing frame PNG files")
    ap.add_argument("--output", required=True, help="Output MP4 file path (e.g., output/video.mp4)")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    args = ap.parse_args()

    frame_paths = list_frame_pngs(args.input)
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in: {args.input}")

    first = cv2.imread(frame_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer for: {args.output}")

    count = 0
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = resize_to_match(frame, width, height)
        writer.write(frame)
        count += 1

    writer.release()
    print(f"Saved {count} frame(s) to: {args.output}")


if __name__ == "__main__":
    main()
