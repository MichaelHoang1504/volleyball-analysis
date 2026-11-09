import json
from pathlib import Path

import cv2
import numpy as np


TEAM_COLORS = {
    "A": (0, 220, 255),   # Yellow
    "B": (255, 180, 0)    # Blue
}


def load_or_select_court(cap, settings_path, video_path):
    """Load saved court areas or let the user select them."""

    settings_path = Path(settings_path)

    success, frame = cap.read()

    if not success:
        raise RuntimeError("Cannot read the court setup frame.")

    # Return to the beginning before processing the video.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    height, width = frame.shape[:2]

    video_info = {
        "video": str(Path(video_path).resolve()),
        "width": width,
        "height": height
    }

    # Reuse settings only for the same video path and dimensions.
    if settings_path.exists():
        data = json.loads(settings_path.read_text())

        if data.get("video_info") == video_info:
            print("Loaded saved court areas.")

            return {
                team: np.array(
                    data["areas"][team],
                    dtype=np.float32
                )
                for team in ("A", "B")
            }

    # Fit the setup image onto the screen.
    scale = min(
        1.0,
        1200 / width,
        700 / height
    )

    display_width = round(width * scale)
    display_height = round(height * scale)

    display_frame = cv2.resize(
        frame,
        (display_width, display_height)
    )

    points = []

    window_name = "Select court: Team A then Team B"

    def mouse_callback(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 8:
                points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Click 4 floor corners for Team A in perimeter order.")
    print("Then click 4 floor corners for Team B.")
    print("ENTER = save | R = reset | ESC = cancel")

    areas = None

    try:
        while True:
            preview = display_frame.copy()

            for index, point in enumerate(points):
                team = "A" if index < 4 else "B"
                color = TEAM_COLORS[team]

                cv2.circle(
                    preview,
                    point,
                    5,
                    color,
                    -1
                )

                cv2.putText(
                    preview,
                    f"{team}{index % 4 + 1}",
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # Connect the selected corners.
            for start, team in ((0, "A"), (4, "B")):
                selected = points[start:start + 4]

                if len(selected) >= 2:
                    polygon = np.array(
                        selected,
                        dtype=np.int32
                    )

                    cv2.polylines(
                        preview,
                        [polygon],
                        len(selected) == 4,
                        TEAM_COLORS[team],
                        2
                    )

            cv2.putText(
                preview,
                "4 corners A, then 4 corners B | ENTER save | R reset",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow(window_name, preview)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                raise RuntimeError("Court selection cancelled.")

            if key in (ord("r"), ord("R")):
                points.clear()

            if key == 13 and len(points) == 8:
                # Convert display coordinates to source coordinates.
                source_points = np.array(
                    points,
                    dtype=np.float32
                )

                source_points[:, 0] *= width / display_width
                source_points[:, 1] *= height / display_height

                areas = {
                    "A": source_points[:4],
                    "B": source_points[4:]
                }

                valid = all(
                    cv2.isContourConvex(polygon)
                    and abs(cv2.contourArea(polygon)) > 50
                    for polygon in areas.values()
                )

                if valid:
                    break

                print(
                    "Invalid selection. Press R and click "
                    "each half's corners in perimeter order."
                )

    finally:
        cv2.destroyWindow(window_name)

    settings_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    settings_path.write_text(
        json.dumps(
            {
                "video_info": video_info,
                "areas": {
                    team: polygon.tolist()
                    for team, polygon in areas.items()
                }
            },
            indent=2
        )
    )

    return areas


def assign_team(box, court_areas):
    """Assign a team using the bottom-center of a person's box."""

    x1, y1, x2, y2 = box

    foot_position = (
        float((x1 + x2) / 2),
        float(y2)
    )

    matching_teams = [
        team
        for team, polygon in court_areas.items()
        if cv2.pointPolygonTest(
            polygon,
            foot_position,
            False
        ) >= 0
    ]

    # Reject people outside both areas or on a shared boundary.
    if len(matching_teams) != 1:
        return None

    return matching_teams[0]


def draw_court_areas(frame, court_areas):
    """Draw thin outlines without covering the court."""

    for team, polygon in court_areas.items():
        cv2.polylines(
            frame,
            [polygon.astype(np.int32)],
            True,
            TEAM_COLORS[team],
            1
        )
