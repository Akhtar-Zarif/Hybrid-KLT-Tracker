# Video Demo (`demo/video_demo.mp4`)

Place your own mission video file (`video_demo_yourname.mp4` or `video demo_yourname.mp4`) inside this `demo/` folder before uploading to GitHub.

## Notes on Uploading Video Files to GitHub
- **GitHub Web UI limit:** Individual files uploaded via the browser interface must be **under 25 MB** (and under **100 MB** via `git push`).
- **Keep it tracked in Git:** The root `.gitignore` is configured to explicitly allow `demo/*.mp4` while ignoring temporary local video exports elsewhere.
- **Embedding in `README.md`:** Once `demo/video_demo.mp4` is pushed to your `main` branch, GitHub automatically renders an inline video player in `README.md` via:
  ```markdown
  https://github.com/Akhtar-Zarif/Hybrid-KLT-Tracker/blob/main/demo/video_demo.mp4
  ```
