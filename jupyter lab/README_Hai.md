# Hai Minh Hoang: setup and pipeline integration

This package covers Hai’s part of the work plan only. It is a new starting version because the current project notebook was not attached. Keep a backup before copying files into your repository.

## Steps
1. Create a branch: `git switch -c hai/pipeline`.
2. Copy `00_setup.ipynb`, `pipeline.ipynb` and `requirements.txt` into the project root. Add the supplied ignore rules to your existing .gitignore; do not overwrite unrelated rules.
3. Open 00_setup.ipynb with your basketball kernel or in Colab. Run the install and GPU cells.
4. Paste the shared Google Drive model and video links. Run the download cells. The links were not included in the plan.
5. Confirm models/ball_yolov8n.pt is the volleyball ball model. Set ball_class in CONFIG if it has more than one class. The person model is separate.
6. Open pipeline.ipynb. Run all cells. Enter your video path and eight real court points in the example cell. Remove the leading # signs from the example, then run it.
7. Open the new outputs folder. Check annotated.mp4 and ball.csv. The times use frame_number / video_fps.
8. Give Ben ball.csv. Empty coordinates mean the ball was missed. These must be ignored before touch detection.
9. Give Anthony run_pipeline_stub and an existing run folder containing all six files. Set config["stub_folder"] to that folder. Return keys are the exact filenames, for example outputs["annotated.mp4"]. Agree on these keys with Anthony.
10. Connect the team functions as they arrive. Empty events and pending stats are placeholders, not final analytics.
11. Re-run the 41-second clip after each merge. Record actual processing time and errors. Do not claim measured results before testing.
12. After the replacement works, use `git mv` to move the exact old main.py/helper files into legacy/. Do not move files while the current code still imports them. Fix the one-byte person weights using the real shared model. Do not commit videos or weights.
13. Run `nbstripout --install` inside the repository. Commit only the notebooks, requirements and README changes. Push the branch and open a pull request for review.

## What is not finished
The final ball smoothing, stable player IDs, jersey teams, action CNN, action labels on the video, ball trail, real statistics, charts and web app belong to later integration. This package does not implement teammates’ research or claim their results.

## Validation
Notebook format and Python syntax were checked. Model inference, the 41-second clip, Google Drive downloads and a fresh Colab run still need your real assets and environment.
