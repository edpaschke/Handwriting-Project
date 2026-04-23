import pathlib
import random
import threading

from flask import Flask, jsonify, render_template, request, send_file

app = Flask(__name__)

BASE_DIR = pathlib.Path(__file__).parent
WRITINGS_DIR = BASE_DIR / "default_writings"
GENERATED_DIR = WRITINGS_DIR / "generated_chars"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

pipeline_status = {"state": "idle", "message": "Upload images and click Process."}


def _run_pipeline():
    global pipeline_status
    try:
        pipeline_status = {"state": "running", "message": "Classifying characters…"}
        from classify_and_store import classify_and_store
        classify_and_store()

        pipeline_status = {"state": "running", "message": "Generating font variations…"}
        from generalize import generalize, save_generated_fonts
        results = generalize()
        save_generated_fonts(results)

        pipeline_status = {
            "state": "done",
            "message": f"Done! Generated {len(results)} characters.",
        }
    except Exception as exc:
        pipeline_status = {"state": "error", "message": str(exc)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    WRITINGS_DIR.mkdir(exist_ok=True)
    saved = []
    for f in files:
        if pathlib.Path(f.filename).suffix.lower() in IMAGE_EXTENSIONS:
            dest = WRITINGS_DIR / pathlib.Path(f.filename).name
            f.save(dest)
            saved.append(f.filename)
    return jsonify({"saved": saved})


@app.route("/process", methods=["POST"])
def process():
    global pipeline_status
    if pipeline_status["state"] == "running":
        return jsonify({"error": "Already running"}), 400
    pipeline_status = {"state": "running", "message": "Starting…"}
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/status")
def status():
    return jsonify(pipeline_status)


@app.route("/chars")
def chars():
    if not GENERATED_DIR.exists():
        return jsonify([])
    return jsonify([d.name for d in sorted(GENERATED_DIR.iterdir()) if d.is_dir()])


@app.route("/char/<path:char_name>")
def char_image(char_name):
    char_dir = GENERATED_DIR / char_name
    if not char_dir.exists():
        return "", 404
    files = list(char_dir.glob("*.png"))
    if not files:
        return "", 404
    return send_file(random.choice(files), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
