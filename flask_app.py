from flask import Flask, render_template, request, send_file, abort, Response, url_for
import os
import cv2
import numpy as np
import csv
import time
import re
from ultralytics import YOLO
from tracker import SimpleTracker

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load models
vehicle_model = YOLO("vehicle_detection.pt")
signal_model = YOLO("signal_violation.pt")
wrongway_model = YOLO("wrong_way_model.pt")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    option = request.form['option']

    if file:
        filename = file.filename
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        if option == 'image':
            frame = cv2.imread(temp_path)
            tracker = SimpleTracker()

            # Detect traffic signal
            signal_results = signal_model.predict(frame, conf=0.5)
            signal_status = 'none'
            for box in signal_results[0].boxes:
                label = signal_model.names[int(box.cls[0])]
                if label in ['red_light', 'green_light']:
                    signal_status = label
                    break

            # Detect vehicles
            vehicle_results = vehicle_model.predict(frame, conf=0.25)
            centroids, boxes = [], []
            for box in vehicle_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                centroids.append(np.array([cx, cy]))
                boxes.append((x1, y1, x2, y2, int(box.cls[0])))

            objects = tracker.update(np.array(centroids))
            vehicle_count = len(objects)
            wrong_way_count = 0

            for i, (object_id, centroid) in enumerate(objects.items()):
                if i < len(boxes):
                    x1, y1, x2, y2, cls = boxes[i]
                    label = vehicle_model.names[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID {object_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Wrong-way detection
            wrong_results = wrongway_model.predict(frame, conf=0.5)
            for box in wrong_results[0].boxes:
                label = wrongway_model.names[int(box.cls[0])]
                if label == 'wrong_way':
                    wrong_way_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "WRONG WAY!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            ts = int(time.time())
            output_image_name = f"output_image_{ts}.jpg"
            output_image_path = os.path.join(PROCESSED_FOLDER, output_image_name)
            cv2.imwrite(output_image_path, frame)

            return render_template('result_image.html',
                                   image_path=url_for('processed', filename=output_image_name),
                                   vehicle_count=vehicle_count,
                                   red_light_violations=1 if signal_status == 'red_light' else 0,
                                   wrong_way_count=wrong_way_count)

        elif option == 'video':
            cap = cv2.VideoCapture(temp_path)
            ret, frame = cap.read()
            height, width = frame.shape[:2]

            ts = int(time.time())
            video_filename = f"output_video_{ts}.mp4"
            csv_filename = f"vehicle_speeds_{ts}.csv"
            out_path = os.path.join(PROCESSED_FOLDER, video_filename)
            csv_path = os.path.join(PROCESSED_FOLDER, csv_filename)

            # Use browser-friendly H.264 codec
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), 15.0, (width, height))

            tracker = SimpleTracker()
            red_light_count = 0
            unique_vehicle_ids = set()
            speed_records = {}
            wrong_way_count = 0
            prev_positions, frame_times = {}, {}
            ppm = 8  # pixels per meter

            while ret:
                signal_results = signal_model.predict(frame, conf=0.5)
                for box in signal_results[0].boxes:
                    label = signal_model.names[int(box.cls[0])]
                    if label == 'red_light':
                        red_light_count += 1

                vehicle_results = vehicle_model.predict(frame, conf=0.25)
                centroids, boxes = [], []
                for box in vehicle_results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centroids.append(np.array([cx, cy]))
                    boxes.append((x1, y1, x2, y2, int(box.cls[0])))

                objects = tracker.update(np.array(centroids))
                for i, (object_id, centroid) in enumerate(objects.items()):
                    unique_vehicle_ids.add(object_id)
                    if i < len(boxes):
                        x1, y1, x2, y2, cls = boxes[i]
                        label = vehicle_model.names[cls]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} ID {object_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        time_now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if object_id in prev_positions:
                            dx = centroid[0] - prev_positions[object_id][0]
                            dy = centroid[1] - prev_positions[object_id][1]
                            dist_px = np.sqrt(dx ** 2 + dy ** 2)
                            time_diff = time_now - frame_times.get(object_id, time_now)
                            if time_diff > 0:
                                dist_m = dist_px / ppm
                                speed_kmph = (dist_m / time_diff) * 3.6
                                cv2.putText(frame, f"{speed_kmph:.1f} km/h", (x1, y2 + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                speed_records.setdefault(object_id, []).append((speed_kmph, time_now))

                        prev_positions[object_id] = centroid
                        frame_times[object_id] = time_now

                wrong_results = wrongway_model.predict(frame, conf=0.5)
                for box in wrong_results[0].boxes:
                    label = wrongway_model.names[int(box.cls[0])]
                    if label == 'wrong_way':
                        wrong_way_count += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, "WRONG WAY!", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                out.write(frame)
                ret, frame = cap.read()

            cap.release()
            out.release()

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Vehicle ID', 'Speed (km/h)', 'Timestamp (s)'])
                for vid, entries in speed_records.items():
                    for speed, t in entries:
                        writer.writerow([vid, round(speed, 2), round(t, 2)])

            all_speeds = [s for v in speed_records.values() for s, _ in v]
            avg_speed = round(np.mean(all_speeds), 2) if all_speeds else 0
            max_speed = round(np.max(all_speeds), 2) if all_speeds else 0

            return render_template('result_video.html',
                                   video_path=url_for('processed', filename=video_filename),
                                   csv_path=url_for('processed', filename=csv_filename),
                                   vehicle_count=len(unique_vehicle_ids),
                                   red_light_violations=red_light_count,
                                   wrong_way_count=wrong_way_count,
                                   avg_speed=avg_speed,
                                   max_speed=max_speed)


@app.route('/processed/<filename>')
def processed(filename):
    filepath = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(filepath):
        abort(404)

    # Streaming support for MP4
    if filename.endswith(".mp4"):
        range_header = request.headers.get('Range', None)
        if not range_header:
            return send_file(filepath, mimetype='video/mp4')

        size = os.path.getsize(filepath)
        byte1, byte2 = 0, None
        match = re.search(r'bytes=(\d+)-(\d*)', range_header)
        if match:
            g = match.groups()
            if g[0]:
                byte1 = int(g[0])
            if g[1]:
                byte2 = int(g[1])

        length = size - byte1
        if byte2 is not None:
            length = byte2 - byte1 + 1

        with open(filepath, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)

        rv = Response(data, status=206, mimetype='video/mp4', direct_passthrough=True)
        rv.headers.add('Content-Range', f'bytes {byte1}-{byte1 + length - 1}/{size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        return rv

    elif filename.endswith(".csv"):
        return send_file(filepath, mimetype='text/csv', as_attachment=True)
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        return send_file(filepath, mimetype='image/jpeg')

    return send_file(filepath)


if __name__ == '__main__':
    app.run(debug=True)
