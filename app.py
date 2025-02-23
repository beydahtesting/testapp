import cv2
import numpy as np
import os
import base64
from flask import Flask, request, render_template, session, redirect, url_for, send_from_directory
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'mysecretkey'  # Change for production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def reorder_points(pts):
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype=np.float32)
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1)
    new_pts[0] = pts[np.argmin(sum_pts)]   # Top-left
    new_pts[1] = pts[np.argmin(diff_pts)]   # Top-right
    new_pts[2] = pts[np.argmax(sum_pts)]    # Bottom-right
    new_pts[3] = pts[np.argmax(diff_pts)]    # Bottom-left
    return new_pts

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            ordered_pts = reorder_points(approx)
            width, height = 1400, 1600
            pts2 = np.float32([[0,0],[width,0],[width,height],[0,height]])
            matrix = cv2.getPerspectiveTransform(ordered_pts, pts2)
            warped_img = cv2.warpPerspective(image, matrix, (width, height))
            return warped_img
    return None

def detect_filled_circles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4*np.pi*(area/(perimeter*perimeter)) if perimeter > 0 else 0
        if 200 < area < 5000 and circularity > 0.6:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            filled_circles.append((x, y, radius))
    return filled_circles

def compare_circles(teacher_circles, student_circles, image):
    threshold = 50
    matched_teacher_indices = []
    matched_student_indices = []
    # Determine matches
    for i, t in enumerate(teacher_circles):
        tx, ty, tr = t
        for j, s in enumerate(student_circles):
            sx, sy, sr = s
            if np.sqrt((tx - sx)**2 + (ty - sy)**2) < threshold:
                matched_teacher_indices.append(i)
                matched_student_indices.append(j)
                break

    graded_image = image.copy()

    # For teacher circles not matched (missed answers), draw outlined green circles
    for i, t in enumerate(teacher_circles):
        tx, ty, tr = t
        if i not in matched_teacher_indices:
            cv2.circle(graded_image, (int(tx), int(ty)), int(tr), (0,255,0), 3)
    # For student circles that match (correct answers), draw filled yellow circles
    for j, s in enumerate(student_circles):
        sx, sy, sr = s
        if j in matched_student_indices:
            cv2.circle(graded_image, (int(sx), int(sy)), int(sr), (0,255,0), -1)
    # For student circles that did not match (wrong selections), draw filled red circles
    for j, s in enumerate(student_circles):
        sx, sy, sr = s
        if j not in matched_student_indices:
            cv2.circle(graded_image, (int(sx), int(sy)), int(sr), (0,0,255), -1)
    return graded_image

# Route to serve processed images from the output folder
@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

# Clear session and start with teacher image
@app.route('/')
def index():
    session.clear()
    return redirect(url_for('upload_key'))

# ----- Teacher Image Upload/Capture -----
@app.route('/upload_key', methods=['GET', 'POST'])
def upload_key():
    if request.method == 'POST':
        teacher_file = request.files.get('teacher')
        if teacher_file:
            teacher_filename = secure_filename(teacher_file.filename)
            teacher_path = os.path.join(app.config['UPLOAD_FOLDER'], teacher_filename)
            teacher_file.save(teacher_path)
            teacher_img = cv2.imread(teacher_path)
            processed = process_image(teacher_img)
            if processed is None:
                return "Error processing teacher image", 400
            teacher_processed = os.path.join(app.config['OUTPUT_FOLDER'], 'teacher_processed.jpg')
            cv2.imwrite(teacher_processed, processed)
            session['teacher_image'] = 'teacher_processed.jpg'
            session['teacher_capture'] = False
            return redirect(url_for('teacher_confirm'))
        # If using capture, the capture endpoint sets session and redirects.
    return render_template('teacher.html')

@app.route('/teacher_confirm')
def teacher_confirm():
    if 'teacher_image' not in session:
        return redirect(url_for('upload_key'))
    return render_template('teacher_confirm.html', teacher_image_url=url_for('output_file', filename=session['teacher_image']))

# ----- Student Image Upload/Capture -----
@app.route('/upload_student', methods=['GET', 'POST'])
def upload_student():
    if 'teacher_image' not in session:
        return redirect(url_for('upload_key'))
    if request.method == 'POST':
        student_file = request.files.get('student')
        if student_file:
            student_filename = secure_filename(student_file.filename)
            student_path = os.path.join(app.config['UPLOAD_FOLDER'], student_filename)
            student_file.save(student_path)
            student_img = cv2.imread(student_path)
            processed = process_image(student_img)
            if processed is None:
                return "Error processing student image", 400
            student_processed = os.path.join(app.config['OUTPUT_FOLDER'], 'student_processed.jpg')
            cv2.imwrite(student_processed, processed)
            session['student_image'] = 'student_processed.jpg'
            session['student_capture'] = False
            return redirect(url_for('student_confirm'))
        # If using capture, the capture endpoint sets session and redirects.
    default_tab = 'capture' if session.get('teacher_capture') else 'upload'
    return render_template('student.html', default_tab=default_tab)

@app.route('/student_confirm')
def student_confirm():
    if 'student_image' not in session:
        return redirect(url_for('upload_student'))
    return render_template('student_confirm.html', student_image_url=url_for('output_file', filename=session['student_image']))

# ----- Grading -----
@app.route('/grade')
def grade():
    if 'teacher_image' not in session or 'student_image' not in session:
        return "Both teacher and student images are required", 400
    teacher_img = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], session['teacher_image']))
    student_img = cv2.imread(os.path.join(app.config['OUTPUT_FOLDER'], session['student_image']))
    teacher_circles = detect_filled_circles(teacher_img)
    student_circles = detect_filled_circles(student_img)
    # Count correct selections (matched student circles)
    correct = 0
    threshold = 50
    for s in student_circles:
        sx, sy, _ = s
        for t in teacher_circles:
            tx, ty, _ = t
            if np.sqrt((sx - tx)**2 + (sy - ty)**2) < threshold:
                correct += 1
                break
    graded = compare_circles(teacher_circles, student_circles, student_img)
    total = len(teacher_circles)
    text = f"{correct} / {total} correct"
    # Add grading text at upper right in red; adjust position if needed
    pos = (graded.shape[1] - 400, 50)
    cv2.putText(graded, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    final_path = os.path.join(app.config['OUTPUT_FOLDER'], 'graded.jpg')
    cv2.imwrite(final_path, graded)
    return render_template('result.html',
                           teacher_image_url=url_for('output_file', filename=session['teacher_image']),
                           student_image_url=url_for('output_file', filename='graded.jpg'))

# ----- Live Capture Endpoint -----
@app.route('/capture', methods=['POST'])
def capture():
    live_data = request.form.get('liveImage')
    image_type = request.form.get('imageType')
    if not live_data or not image_type:
        return "Missing image data", 400
    header, encoded = live_data.split(',', 1)
    data = base64.b64decode(encoded)
    pil_img = Image.open(BytesIO(data))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    processed = process_image(img)
    if processed is None:
        return "Error processing captured image", 400
    if image_type == 'teacher':
        result_path = os.path.join(app.config['OUTPUT_FOLDER'], 'teacher_processed.jpg')
        cv2.imwrite(result_path, processed)
        session['teacher_image'] = 'teacher_processed.jpg'
        session['teacher_capture'] = True
        return {"redirect": url_for('teacher_confirm')}
    elif image_type == 'student':
        result_path = os.path.join(app.config['OUTPUT_FOLDER'], 'student_processed.jpg')
        cv2.imwrite(result_path, processed)
        session['student_image'] = 'student_processed.jpg'
        session['student_capture'] = True
        return {"redirect": url_for('student_confirm')}
    else:
        return "Invalid image type", 400

# Option to start over with a new key
@app.route('/new_key')
def new_key():
    session.clear()
    return redirect(url_for('upload_key'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
