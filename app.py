import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, Response
import processing as p
import time

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
HIST_FOLDER = "static/hist"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HIST_FOLDER, exist_ok=True)

# ---------------- IMAGE PROCESSING ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    before = None
    after = None
    hist = None

    if request.method == "POST":
        file = request.files["image"]
        if not file or file.filename == "":
         return render_template("index.html")
        operation = request.form["operation"]

        blur_val = int(request.form.get("blur", 5))
        thresh_val = int(request.form.get("thresh", 127))

        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            img = cv2.imread(path)
            if img is None:
             print("❌ Image not loaded")
            else:
             print("✅ Image loaded successfully")

            # Processing
            if operation == "blur":
                k = blur_val if blur_val % 2 else blur_val + 1
                result = cv2.GaussianBlur(img, (k, k), 0)

            elif operation == "threshold":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, result = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            elif operation == "enhance":
                result = p.auto_enhance(img)

            elif operation == "detect":
                result = p.detect_objects(img)

            else:
                result = img

            filename = str(int(time.time())) + ".png"
            result_path = os.path.join(RESULT_FOLDER, filename)
            cv2.imwrite(result_path, result)

            # Histogram
            hist_filename = str(int(time.time())) + "_hist.png"
            hist_path = os.path.join(HIST_FOLDER, hist_filename)

            plt.figure()
            if len(result.shape) == 2:
                plt.hist(result.ravel(), bins=256)
            else:
                colors = ('b','g','r')
                for i, col in enumerate(colors):
                    hist_data = cv2.calcHist([result],[i],None,[256],[0,256])
                    plt.plot(hist_data, color=col)

            plt.title("Histogram")
            plt.savefig(hist_path)
            plt.close()

            before = path
            after = result_path
            hist = hist_path

    return render_template("index.html", before=before, after=after, hist=hist)

# ---------------- LIVE CAMERA ----------------
# ---------------- LIVE CAMERA ----------------

cap = None
camera_on = False

def generate_frames():
    global cap, camera_on

    if cap is None:
        cap = cv2.VideoCapture(0)

    while camera_on:
        success, frame = cap.read()
        if not success:
            break

        processed = cv2.Canny(frame, 100, 200)

        ret, buffer = cv2.imencode('.jpg', processed)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera_on
    camera_on = True
    return ("", 204)

@app.route('/stop_camera')
def stop_camera():
    global camera_on, cap
    camera_on = False
    if cap:
        cap.release()
        cap = None
    return ("", 204)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)