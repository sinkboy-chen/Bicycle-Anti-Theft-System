def send_gmail_blackout(img_path):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    GMAIL_USER = os.environ.get("GMAIL_USER")
    GMAIL_PASS = os.environ.get("GMAIL_PASS")
    subject = "警告: 攝影機疑似被遮蔽!"
    body = "自動警告：攝影機畫面長時間過暗，可能遭遮蔽或破壞，請立即檢查現場安全。"
    to_email = "ernestii260928@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg.set_content(body, subtype="plain", charset="utf-8")

    with open(img_path, "rb") as f:
        file_data = f.read()
        file_name = Path(img_path).name
    msg.add_attachment(
        file_data,
        maintype="image",
        subtype="jpeg",
        filename=(Header(file_name, "utf-8").encode())
    )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, Response, render_template_string
import threading
import time
import os
import smtplib
from email.message import EmailMessage
from email.header import Header
from pathlib import Path
def send_gmail(img_path):
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    GMAIL_USER = os.environ.get("GMAIL_USER")
    GMAIL_PASS = os.environ.get("GMAIL_PASS")
    print(GMAIL_USER, GMAIL_PASS)
    subject = "警告: 非主人臉部偵測"
    body = "這是一封自動警告通知，偵測到非主人的臉部。請查閱附件圖片。"
    to_email = "ernestii260928@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = str(Header(subject, "utf-8"))
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg.set_content(body, subtype="plain", charset="utf-8")

    with open(img_path, "rb") as f:
        file_data = f.read()
        file_name = Path(img_path).name
    msg.add_attachment(
        file_data,
        maintype="image",
        subtype="jpeg",
        filename=(Header(file_name, "utf-8").encode())
    )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)

# Paths
FACE_DETECT_MODEL = os.path.join(os.path.dirname(__file__), 'yolo_face_detect.tflite')
FACE_RECOG_MODEL = os.path.join(os.path.dirname(__file__), 'facenet512_uint8.tflite')
OWNER_IMG_PATH = os.path.join(os.path.dirname(__file__), 'owner.jpg')

# Load owner embedding
owner_embedding = None

def get_face_embedding(face_img, recog_interpreter, input_scale, input_zero_point):
    # Preprocess face for facenet512_uint8.tflite
    img = cv2.resize(face_img, (160, 160))
    img = img[None, ...] / 255.0
    img = (img / input_scale + input_zero_point).astype(np.uint8)
    input_details = recog_interpreter.get_input_details()
    recog_interpreter.set_tensor(input_details[0]['index'], img)
    recog_interpreter.invoke()
    output_details = recog_interpreter.get_output_details()
    embedding = recog_interpreter.get_tensor(output_details[0]['index'])[0]
    return embedding

def cosine_similarity(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_owner_embedding(recog_interpreter, input_scale, input_zero_point):
    global owner_embedding
    owner_img = cv2.imread(OWNER_IMG_PATH)
    if owner_img is None:
        raise FileNotFoundError('owner.jpg not found in publish folder')
    owner_embedding = get_face_embedding(owner_img, recog_interpreter, input_scale, input_zero_point)

# Flask app
app = Flask(__name__)
latest_frame = None
latest_non_owner_faces = []
blackout_countdown = 0

@app.route('/')
def index():
    global blackout_countdown
    return render_template_string('''
        <h1>Camera Feed</h1>
        <img src="/video_feed" width="480"/><br>
        {% if blackout_countdown > 0 %}
            <div style="color:red;font-size:1.5em;">
                <b>警告：攝影機畫面過暗，{{ blackout_countdown }} 秒後將發送遮蔽警報！</b>
            </div>
        {% endif %}
        <h2>Non-owner Faces</h2>
        {% for idx in range(non_owner_count) %}
            <img src="/non_owner_face/{{idx}}" width="160"/>
        {% endfor %}
    ''', non_owner_count=len(latest_non_owner_faces), blackout_countdown=blackout_countdown)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if latest_frame is not None:
                _, jpeg = cv2.imencode('.jpg', latest_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/non_owner_face/<int:idx>')
def non_owner_face(idx):
    if 0 <= idx < len(latest_non_owner_faces):
        _, jpeg = cv2.imencode('.jpg', latest_non_owner_faces[idx])
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return '', 404

def camera_loop():
    global latest_frame, latest_non_owner_faces
    global blackout_countdown
    BLACKOUT_THRESHOLD = 40  # average pixel value below this is considered black
    BLACKOUT_SECONDS = 10
    blackout_start = None
    blackout_countdown = 0
    # Load TFLite models
    face_interpreter = tflite.Interpreter(model_path=FACE_DETECT_MODEL)
    face_interpreter.allocate_tensors()
    recog_interpreter = tflite.Interpreter(model_path=FACE_RECOG_MODEL)
    recog_interpreter.allocate_tensors()
    recog_input_details = recog_interpreter.get_input_details()
    input_scale, input_zero_point = recog_input_details[0]["quantization"]
    load_owner_embedding(recog_interpreter, input_scale, input_zero_point)

    face_input_details = face_interpreter.get_input_details()
    face_output_details = face_interpreter.get_output_details()
    face_input_shape = face_input_details[0]["shape"] # e.g. [1, 128, 128, 3]
    face_input_scale, face_input_zero_point = face_input_details[0]["quantization"]
    face_input_dtype = face_input_details[0]["dtype"] if "dtype" in face_input_details[0] else np.int8

    def image_resize(ori_img, dst_shape):
        img_shape = ori_img.shape
        W, H = dst_shape
        scale = min(W / img_shape[1], H / img_shape[0])
        nw = int(img_shape[1] * scale)
        nh = int(img_shape[0] * scale)
        dx = (W - nw) // 2
        dy = (H - nh) // 2
        res_img = cv2.resize(ori_img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        new_img = np.ones((H, W, 3), np.uint8) * 128
        for i in range(nh):
            for j in range(nw):
                new_img[dy + i][dx + j] = res_img[i][j]
        return new_img

    # Anchors for face detection (from example_lite_detection.py)
    anchors = [
        [[51, 64], [59, 82], [79, 100]],
        [[29, 51], [36, 43], [41, 54]],
        [[15, 21], [22, 29], [28, 36]],
    ]
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    def nonMaxSuppress(boxLoc, score, maxBox=20, iouThresh=0.5):
        x1 = boxLoc[:, 0]
        y1 = boxLoc[:, 1]
        x2 = boxLoc[:, 2]
        y2 = boxLoc[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = score.argsort()[::-1]
        keep = []
        while order.size > 0 and len(keep) < maxBox:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iouThresh)[0]
            order = order[inds + 1]
        return boxLoc[keep], score[keep]
    def decode(netout, anchors, net_shape, image_shape, conf_thres=0.7, nms_score=0.45):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        net_h, net_w = net_shape[1:3]
        image_h, image_w = image_shape[:2]
        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        if (float(net_w) / image_w) < (float(net_h) / image_h):
            new_w = net_w
            new_h = (image_h * net_w) / image_w
        else:
            new_h = net_w
            new_w = (image_w * net_h) / image_h
        x_offset, x_scale = (net_w - new_w) / 2.0 / net_w, float(net_w) / new_w
        y_offset, y_scale = (net_h - new_h) / 2.0 / net_h, float(net_h) / new_h
        cell_scores = netout[..., 4]
        cell_scores = np.expand_dims(cell_scores, axis=-1)
        col = np.array([[j for j in range(grid_w)] for _ in range(grid_h)])
        row = np.array([[i for _ in range(grid_w)] for i in range(grid_h)])
        col = np.reshape(col, newshape=(col.shape[0], col.shape[1], 1, 1))
        row = np.reshape(row, newshape=(row.shape[0], row.shape[1], 1, 1))
        x = np.expand_dims(netout[..., 0], axis=-1)
        y = np.expand_dims(netout[..., 1], axis=-1)
        w = np.expand_dims(netout[..., 2], axis=-1)
        h = np.expand_dims(netout[..., 3], axis=-1)
        x = (col + x) / grid_w
        y = (row + y) / grid_h
        anchors = np.expand_dims(anchors, axis=(0, 1))
        anchors_w = np.expand_dims(anchors[..., 0], axis=-1)
        anchors_h = np.expand_dims(anchors[..., 1], axis=-1)
        w = anchors_w * np.exp(w) / net_w
        h = anchors_h * np.exp(h) / net_h
        classes = netout[..., 5:]
        x = (x - x_offset) * x_scale
        y = (y - y_offset) * y_scale
        w *= x_scale
        h *= y_scale
        x2 = (x + w / 2) * image_w
        y2 = (y + h / 2) * image_h
        x2 = np.where(x2 >= image_w, image_w - 10, x2)
        y2 = np.where(y2 >= image_h, image_h - 10, y2)
        boxes = np.concatenate(
            [(x - w / 2) * image_w, (y - h / 2) * image_h, x2, y2, cell_scores, classes],
            axis=-1,
        )
        boxes_coord = boxes[..., :4]
        cell_scores = np.squeeze(cell_scores, axis=-1)
        mask = cell_scores > conf_thres
        boxes_over_thresh = boxes_coord[mask]
        scores_over_thresh = cell_scores[mask]
        if boxes_over_thresh.shape[0] == 0:
            return []
        selected_boxes, selected_score = nonMaxSuppress(
            boxes_over_thresh, scores_over_thresh, iouThresh=nms_score
        )
        selected_score = np.expand_dims(selected_score, axis=1)
        found_objects = np.concatenate((selected_boxes, selected_score), axis=1)
        return found_objects

    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    import tempfile
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        logging.info('Got image')
        # Blackout detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness < BLACKOUT_THRESHOLD:
            if blackout_start is None:
                blackout_start = time.time()
            blackout_countdown = max(0, BLACKOUT_SECONDS - int(time.time() - blackout_start))
            logging.info(f"Camera blackout detected. Countdown: {blackout_countdown}s (brightness={avg_brightness:.1f})")
            if time.time() - blackout_start >= BLACKOUT_SECONDS:
                # Send blackout alert email
                blackout_img_path = os.path.join(os.path.dirname(__file__), 'blackout.jpg')
                cv2.imwrite(blackout_img_path, frame)
                try:
                    send_gmail_blackout(blackout_img_path)
                    logging.info("Sent blackout alert email.")
                except Exception as e:
                    logging.error(f"Failed to send blackout Gmail: {e}")
                blackout_start = None
                blackout_countdown = 0
            # Skip face detection if blackout
            latest_frame = frame.copy()
            latest_non_owner_faces = []
            time.sleep(0.05)
            continue
        else:
            blackout_start = None
            blackout_countdown = 0
        # ...existing code for face detection and recognition...
        logging.info('Performing detection')
        _, H, W, C = face_input_shape
        resized = image_resize(frame, (W, H))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.reshape(face_input_shape)
        img = img / 255.0
        if face_input_dtype == np.int8 or (isinstance(face_input_dtype, str) and 'int8' in face_input_dtype):
            img = (img / face_input_scale + face_input_zero_point).astype(np.int8)
        else:
            img = (img / face_input_scale + face_input_zero_point).astype(np.uint8)
        face_interpreter.set_tensor(face_input_details[0]['index'], img)
        face_interpreter.invoke()
        # Postprocess output using anchors and decode
        boxes = []
        for i in range(len(face_output_details)):
            anchor = anchors[i]
            out = face_interpreter.get_tensor(face_output_details[i]['index'])[0]
            zero_point = face_output_details[i]["quantization_parameters"]["zero_points"]
            scale = face_output_details[i]["quantization_parameters"]["scales"]
            out = ((out - zero_point) * scale).astype("float32")
            box = decode(out, anchor, face_input_shape, frame.shape)
            if len(box) > 0:
                boxes.append(box)
        if len(boxes) == 0:
            found_objects = []
        else:
            found_objects = np.concatenate(boxes, axis=0)
        logging.info(f'Detected {len(found_objects)} faces')
        h, w, _ = frame.shape
        non_owner_faces = []
        for idx, box in enumerate(found_objects):
            x1, y1, x2, y2 = [int(box[i]) for i in range(4)]
            if x2-x1 < 10 or y2-y1 < 10:
                continue
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            logging.info(f'Recognizing face {idx+1}')
            embedding = get_face_embedding(face_img, recog_interpreter, input_scale, input_zero_point)
            sim = cosine_similarity(owner_embedding, embedding)
            if sim <= 0.4:
                logging.info(f"Owner's face detected (score={sim:.3f})")
            else:
                logging.info(f"Not owner's face detected (score={sim:.3f})")
                non_owner_faces.append(face_img)
                # Send non-owner face via Gmail
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    cv2.imwrite(tmp.name, face_img)
                    try:
                        send_gmail(tmp.name)
                        logging.info(f"Sent non-owner face via Gmail: {tmp.name}")
                    except Exception as e:
                        logging.error(f"Failed to send Gmail: {e}")
            # Draw box
            color = (0,255,0) if sim <= 0.4 else (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        latest_frame = frame.copy()
        latest_non_owner_faces = non_owner_faces
        time.sleep(0.05)

def main():
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
