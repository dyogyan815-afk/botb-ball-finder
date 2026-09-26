# BOTB Ball Finder

Research-only human-judge coordinate emulator. It accepts an image and returns a learned coordinate estimate plus uncertainty. It does not click, submit entries, purchase tickets, scrape BOTB, or automate the BOTB site.

It cannot guarantee 100% accuracy. Train it on independently annotated images before using predictions.

## Run backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`POST /predict` with multipart field `image` returns coordinates in the uploaded image's pixel coordinate system.

## Train

Create `data/annotations.jsonl`, one record per image:

```json
{"image_path":"data/images/a.jpg","x":742,"y":391}
```

Then run:

```bash
python -m app.train --annotations data/annotations.jsonl --output checkpoints/model.pt --epochs 20
```

The initial model is a baseline, not a guaranteed solver. Evaluate on a held-out image-level split and compare against human-human disagreement.

## Local userscript

Install `userscript/botb-local-panel.user.js` in Tampermonkey/Userscripts. It adds a local upload panel only; it does not modify or submit the BOTB page. Start the backend first.

## iOS

Open `ios/BallFinder/BallFinder.xcodeproj` after creating an Xcode iOS App project, or copy `ContentView.swift` and `BallFinderApp.swift` into a SwiftUI project. Set the API URL to the computer running the backend (for a physical phone, use the computer's LAN IP, not localhost).

## License

MIT
