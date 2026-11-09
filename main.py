import csv
from collections import deque
from pathlib import Path
from court_utils import (
    TEAM_COLORS,
    load_or_select_court,
    assign_team,
    draw_court_areas
)
import cv2
import numpy as np
from ultralytics import YOLO

from helper import create_video_writer


def main():
    # --------------------------------------------------
    # 1. File paths
    # --------------------------------------------------
    # Resolve paths relative to this Python file.
    project_folder = Path(__file__).resolve().parent

    input_path = project_folder / "volleyball_sample.mp4"

    output_folder = project_folder / "outputs"
    output_folder.mkdir(parents=True, exist_ok=True)

    run_number = 1

    # Find the next available number for both output files.
    while True:
        output_path = (
            output_folder / f"volleyball_tracking_{run_number}.mp4"
        )

        csv_path = (
            output_folder / f"player_tracks_{run_number}.csv"
        )

        if not output_path.exists() and not csv_path.exists():
            break

        run_number += 1

    # Existing volleyball-trained ball detector.
    ball_model_path = project_folder / "yolov8n.pt"

    # Separate general detector for people.
    player_model_path = (
        project_folder / "player_models" / "yolov8n.pt"
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    if not ball_model_path.is_file():
        raise FileNotFoundError(
            f"Volleyball model not found: {ball_model_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_model_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Load models ONCE, before processing frames
    # --------------------------------------------------
    ball_model = YOLO(str(ball_model_path))

    # Ultralytics downloads standard weights here
    # if this file does not already exist.
    player_model = YOLO(str(player_model_path))

    print("Ball model classes:", ball_model.names)
    print("Player model classes:", player_model.names)

    # Identify ball classes using the model's class names.
    ball_class_ids = [
        class_id
        for class_id, name in ball_model.names.items()
        if "ball" in name.lower()
    ]

    if not ball_class_ids:
        raise ValueError(
            "The ball model has no class containing 'ball'. "
            "Check that yolov8n.pt is your volleyball-trained model."
        )

    # Standard COCO YOLO weights use class 0 for person.
    if player_model.names.get(0, "").lower() != "person":
        raise ValueError(
            "The player model is incorrect. "
            "player_models/yolov8n.pt must be the standard "
            "COCO model, not your volleyball-trained model."
        )

    # --------------------------------------------------
    # 3. Open video
    # --------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open the input video: {input_path}"
        )
    court_areas = load_or_select_court(
        cap,
        project_folder / "court_areas.json",
        input_path
    )
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(cap, output_path)

    # --------------------------------------------------
    # 4. Tracking state
    # --------------------------------------------------
    frame_number = 0

    ball_trail = deque(maxlen=10)
    previous_ball = None
    missed_ball_frames = 0

    # Forget an old ball position after about 0.3 seconds.
    max_missed_frames = max(1, round(fps * 0.3))

    # Approximate maximum displacement between frames.
    # This is a starting heuristic, not a physical limit.
    max_ball_jump = (
        0.25 * np.hypot(frame_width, frame_height)
    )

    # Scale labels to the source video resolution.
    font_scale = max(0.4, frame_width / 1800)

    # --------------------------------------------------
    # 5. Create CSV and process video
    # --------------------------------------------------
    try:
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame",
                "time_seconds",
                "player_id",
                "team",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2"
            ])

            while True:
                success, frame = cap.read()

                if not success:
                    break

                # Both models receive the original frame.
                # Draw annotations on a separate copy.
                annotated_frame = frame.copy()

                time_seconds = frame_number / fps

                # --------------------------------------
                # A. Detect, track and assign teams
                # --------------------------------------
                player_result = player_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False
                )[0]

                player_boxes = player_result.boxes

                tracked_player_count = 0
                team_counts = {"A": 0, "B": 0}

                draw_court_areas(
                    annotated_frame,
                    court_areas
                )

                if (
                    player_boxes is not None
                    and player_boxes.id is not None
                ):
                    coordinates = (
                        player_boxes.xyxy.cpu().numpy()
                    )

                    player_ids = (
                        player_boxes.id.int().cpu().tolist()
                    )

                    confidences = (
                        player_boxes.conf.cpu().tolist()
                    )

                    for box, player_id, confidence in zip(
                        coordinates,
                        player_ids,
                        confidences
                    ):
                        team = assign_team(
                            box,
                            court_areas
                        )

                        # Skip people outside the selected court areas.
                        if team is None:
                            continue

                        tracked_player_count += 1
                        team_counts[team] += 1

                        x1, y1, x2, y2 = map(int, box)

                        color = TEAM_COLORS[team]

                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1),
                            (x2, y2),
                            color,
                            2
                        )

                        cv2.putText(
                            annotated_frame,
                            f"Team {team} | ID {player_id}",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            color,
                            1,
                            cv2.LINE_AA
                        )

                        # Show the point used for court assignment.
                        foot_position = (
                            int((box[0] + box[2]) / 2),
                            int(box[3])
                        )

                        cv2.circle(
                            annotated_frame,
                            foot_position,
                            4,
                            color,
                            -1
                        )

                        csv_writer.writerow([
                            frame_number,
                            round(time_seconds, 3),
                            player_id,
                            team,
                            round(confidence, 3),
                            x1,
                            y1,
                            x2,
                            y2
                        ])

                # --------------------------------------
                # B. Detect the volleyball
                # --------------------------------------
                ball_result = ball_model.predict(
                    frame,
                    classes=ball_class_ids,
                    conf=0.5,
                    verbose=False
                )[0]

                candidates = []

                if ball_result.boxes is not None:
                    coordinates = (
                        ball_result.boxes.xyxy.cpu().numpy()
                    )

                    confidences = (
                        ball_result.boxes.conf.cpu().tolist()
                    )

                    for box, confidence in zip(
                        coordinates,
                        confidences
                    ):
                        x1, y1, x2, y2 = box

                        center = np.array([
                            (x1 + x2) / 2,
                            (y1 + y2) / 2
                        ])

                        candidates.append(
                            (center, confidence)
                        )

                # --------------------------------------
                # C. Select one ball candidate
                # --------------------------------------
                selected_ball = None

                if candidates:
                    if previous_ball is None:
                        # No previous position:
                        # choose the most confident detection.
                        selected_ball = max(
                            candidates,
                            key=lambda item: item[1]
                        )[0]

                    else:
                        # Choose the detection closest
                        # to the previous ball position.
                        closest = min(
                            candidates,
                            key=lambda item: np.linalg.norm(
                                item[0] - previous_ball
                            )
                        )[0]

                        distance = np.linalg.norm(
                            closest - previous_ball
                        )

                        if distance <= max_ball_jump:
                            selected_ball = closest

                # --------------------------------------
                # D. Update and draw the ball trail
                # --------------------------------------
                if selected_ball is not None:
                    previous_ball = selected_ball
                    missed_ball_frames = 0

                    center = tuple(
                        int(value) for value in selected_ball
                    )

                    ball_trail.append(center)

                    cv2.circle(
                        annotated_frame,
                        center,
                        8,
                        (255, 0, 0),
                        2
                    )

                    points = list(ball_trail)

                    for start, end in zip(
                        points[:-1],
                        points[1:]
                    ):
                        cv2.line(
                            annotated_frame,
                            start,
                            end,
                            (0, 0, 255),
                            2
                        )

                else:
                    missed_ball_frames += 1

                    # Do not draw a trail across missing frames.
                    ball_trail.clear()

                    if missed_ball_frames >= max_missed_frames:
                        previous_ball = None

                # --------------------------------------
                # E. Draw a compact information panel
                # --------------------------------------
                cv2.rectangle(
                    annotated_frame,
                    (0, 0),
                    (frame_width, 36),
                    (25, 25, 25),
                    -1
                )

                cv2.putText(
                    annotated_frame,
                    (
                        f"Time: {time_seconds:.1f}s | "
                        f"Team A: {team_counts['A']} | "
                        f"Team B: {team_counts['B']}"
                    ),
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # --------------------------------------
                # F. Save annotated frame
                # --------------------------------------
                writer.write(annotated_frame)

                frame_number += 1

                if frame_number % 50 == 0:
                    print(
                        f"Processed {frame_number} frames",
                        flush=True
                    )

    finally:
        cap.release()
        writer.release()

    print(f"Finished processing {frame_number} frames.")
    print(f"Output video: {output_path}")
    print(f"Player tracking CSV: {csv_path}")


if __name__ == "__main__":
    main()