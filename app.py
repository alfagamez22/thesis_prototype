from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for, flash
import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'har'))
import backend.har.livefeed as livefeed
from functools import wraps
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
import backend.models as models
from backend.models import db, Employee
from backend.config import Config
from threading import Lock
import threading
import time

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Create database tables
with app.app_context():
    db.create_all()

# Start the background video thread when the app starts
livefeed.start_background_video_thread()

# Global dictionary to store post-processing progress
postprocess_progress = {}
postprocess_progress_lock = Lock()

# Utility to update progress (to be called from post-processing logic)
def set_postprocess_progress(recording_id, percent, status=None):
    with postprocess_progress_lock:
        postprocess_progress[recording_id] = {
            'percent': percent,
            'status': status or '',
        }

def get_postprocess_progress(recording_id):
    with postprocess_progress_lock:
        return postprocess_progress.get(recording_id, {'percent': 0, 'status': 'Queued'})

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Replace this with your actual user authentication logic
        if username == "username" and password == "password":  # Example credentials
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route("/")
@login_required
def home():
    return render_template("home.html")

@app.route("/about")
@login_required
def about():
    return render_template("about.html")

@app.route("/livefeed")
@login_required
def livefeed_page():
    return render_template("livefeed.html")

@app.route("/recordings")
@login_required
def recordings():
    return render_template("recordings.html")

@app.route("/employees")
@login_required
def employees():
    return render_template("employees.html")

@app.route("/account")
@login_required
def account():
    # For demonstration, use a mock user or fetch from session/db if available
    user = None
    if session.get('user_id'):
        from backend.models import User
        user = User.query.get(session['user_id'])
    if not user:
        # fallback: show a demo user (since login is hardcoded)
        user = type('User', (), {
            'username': 'username',
            'email': 'demo@email.com',
            'role': 'user',
            'full_name': 'Demo User',
            'created_at': None,
            'last_login': None
        })()
    return render_template("account.html", user=user)

@app.route("/employee_activity")
@login_required
def employee_activity():
    return render_template("employee_activity.html")

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = livefeed.get_latest_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                # Short sleep to prevent CPU spinning when no frames available
                import time
                time.sleep(0.01)  # Reduced from 0.05 to make it more responsive
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/activity_status')
def activity_status():
    return jsonify({"active": livefeed.get_activity_recognition_state()})

@app.route('/toggle_activity', methods=['POST'])
def toggle_activity():
    new_state = livefeed.toggle_activity_recognition()
    return jsonify({"active": new_state})

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    data = request.get_json()
    recording_type = data.get('type', 'original') if data else 'original'
    
    current_status = livefeed.get_recording_status()
    
    if recording_type == 'original':
        if not current_status['original']:
            livefeed.start_recording('original')
            return jsonify({"recording": True, "type": "original"})
        else:
            livefeed.stop_recording('original')
            return jsonify({"recording": False, "type": "original"})
    else:  # segmented
        if not current_status['segmented']:
            livefeed.start_recording('segmented')
            return jsonify({"recording": True, "type": "segmented"})
        else:
            livefeed.stop_recording('segmented')
            return jsonify({"recording": False, "type": "segmented"})

@app.route('/toggle_dual_recording', methods=['POST'])
def toggle_dual_recording():
    """Toggle both original and segmented recording simultaneously"""
    current_status = livefeed.get_recording_status()
    
    # If either recording is active, stop both
    if current_status['original'] or current_status['segmented']:
        livefeed.stop_dual_recording()
        return jsonify({"recording": False, "message": "Both recordings stopped"})
    else:
        # Start both recordings
        success = livefeed.start_dual_recording()
        return jsonify({"recording": success, "message": "Both recordings started" if success else "Failed to start recordings"})

@app.route('/recording_status')
def recording_status():
    status = livefeed.get_recording_status()
    return jsonify(status)

@app.route('/download_recording/<filename>')
def download_recording(filename):
    return send_from_directory('backend','recordings', filename, as_attachment=True)

# @app.route('/list_recordings')
# def list_recordings():
#     recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
#     if not os.path.exists(recordings_dir):
#         os.makedirs(recordings_dir)
    
#     files = [f for f in os.listdir(recordings_dir) if f.endswith('.mp4')]
#     return jsonify({"files": files})

# @app.route('/recordings/<filename>')
# def serve_recording(filename):
#     recordings_dir = os.path.join(os.path.dirname(__file__), 'recordings')
#     return send_from_directory(recordings_dir, filename)

@app.route('/list_recordings')
def list_recordings():
    # Always use dynamic, portable paths
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    recordings_dir = os.path.join(backend_dir, 'recordings')
    segmented_dir = os.path.join(recordings_dir, 'segmented_recordings')

    files = []
    processing_files = []

    # Get original recordings (excluding raw segmented files)
    if os.path.exists(recordings_dir):
        original_files = [f for f in os.listdir(recordings_dir)
                         if f.endswith(('.mp4', '.avi'))
                         and os.path.isfile(os.path.join(recordings_dir, f))
                         and not f.startswith('raw_segmented_recording_')]
        files.extend(original_files)

        # Check for raw segmented files that are being processed
        raw_files = [f for f in os.listdir(recordings_dir)
                    if f.startswith('raw_segmented_recording_') and f.endswith(('.mp4', '.avi'))]
        processing_files.extend(raw_files)

    # Get segmented recordings
    if os.path.exists(segmented_dir):
        segmented_files = [f for f in os.listdir(segmented_dir)
                          if f.endswith(('.mp4', '.avi'))]
        # Add path prefix to distinguish segmented recordings
        files.extend([f"segmented_recordings/{f}" for f in segmented_files])

    return jsonify({
        "files": files,
        "processing": processing_files  # Files currently being processed
    })

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    import mimetypes
    recordings_dir = os.path.join(os.path.dirname(__file__),'backend', 'recordings')
    
    # Handle both direct files and segmented_recordings subdirectory
    if filename.startswith('segmented_recordings/'):
        # Remove the prefix and look in segmented_recordings subdirectory
        actual_filename = filename.replace('segmented_recordings/', '')
        file_path = os.path.join(recordings_dir, 'segmented_recordings', actual_filename)
        serve_dir = os.path.join(recordings_dir, 'segmented_recordings')
        serve_filename = actual_filename
    else:
        # Look in main recordings directory
        file_path = os.path.join(recordings_dir, filename)
        serve_dir = recordings_dir
        serve_filename = filename
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Get file info
    file_size = os.path.getsize(file_path)
    
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(serve_filename)
    if not mime_type:
        if serve_filename.lower().endswith('.mp4'):
            mime_type = 'video/mp4'
        elif serve_filename.lower().endswith('.avi'):
            mime_type = 'video/x-msvideo'
        else:
            mime_type = 'application/octet-stream'
    
    print(f"Serving video: {filename}, size: {file_size}, mime: {mime_type}")
    
    return send_from_directory(
        serve_dir, 
        serve_filename,
        mimetype=mime_type,
        as_attachment=False,
        conditional=True  # Enable conditional requests for video streaming
    )

@app.route('/delete_recording/<path:filename>', methods=['DELETE'])
def delete_recording(filename):
    try:
        recordings_dir = os.path.join(os.path.dirname(__file__),'backend', 'recordings')
        
        # Handle both direct files and segmented_recordings subdirectory
        if filename.startswith('segmented_recordings/'):
            # Remove the prefix and look in segmented_recordings subdirectory
            actual_filename = filename.replace('segmented_recordings/', '')
            file_path = os.path.join(recordings_dir, 'segmented_recordings', actual_filename)
        else:
            # Look in main recordings directory
            file_path = os.path.join(recordings_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "File not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Employee CRUD routes
@app.route('/api/employees', methods=['GET'])
@login_required
def get_employees():
    employees = Employee.query.all()
    return jsonify([employee.to_dict() for employee in employees])

@app.route('/api/employees/<int:id>', methods=['GET'])
@login_required
def get_employee(id):
    employee = Employee.query.get_or_404(id)
    return jsonify(employee.to_dict())

@app.route('/api/employees', methods=['POST'])
@login_required
def create_employee():
    data = request.json
    
    # Convert hire_date string to date object
    hire_date = datetime.strptime(data['hireDate'], '%Y-%m-%d').date()
    
    employee = Employee(
        employee_id=data['employeeId'],
        full_name=data['fullName'],
        photo_url=data.get('photoUrl'),
        role=data['role'],
        hire_date=hire_date,
        email=data['email'],
        phone=data.get('phone'),
        status=data['status'],
        department=data.get('department')
    )
    
    try:
        db.session.add(employee)
        db.session.commit()
        return jsonify(employee.to_dict()), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/employees/<int:id>', methods=['PUT'])
@login_required
def update_employee(id):
    employee = Employee.query.get_or_404(id)
    data = request.json

    # Map camelCase keys to snake_case for model attributes
    key_map = {
        'employeeId': 'employee_id',
        'fullName': 'full_name',
        'photoUrl': 'photo_url',
        'hireDate': 'hire_date',
        'department': 'department',
        'role': 'role',
        'email': 'email',
        'phone': 'phone',
        'status': 'status',
    }

    for camel, snake in key_map.items():
        if camel in data:
            value = data[camel]
            if snake == 'hire_date' and value:
                try:
                    value = datetime.strptime(value, '%Y-%m-%d').date()
                except Exception:
                    return jsonify({'error': 'Invalid hire date format'}), 400
            setattr(employee, snake, value)

    try:
        db.session.commit()
        return jsonify(employee.to_dict())
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/api/employees/<int:id>', methods=['DELETE'])
@login_required
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    
    try:
        db.session.delete(employee)
        db.session.commit()
        return '', 204
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400

@app.route('/capture_employee_activity', methods=['POST'])
@login_required
def capture_employee_activity():
    try:
        result = livefeed.capture_employee_activity()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/list_employee_captures')
@login_required
def list_employee_captures():
    try:
        result = livefeed.get_employee_captures()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/clear_employee_captures', methods=['DELETE'])
@login_required
def clear_employee_captures():
    try:
        result = livefeed.clear_employee_captures()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/employee_captures/<path:filename>')
@login_required
def serve_employee_capture(filename):
    # Serve from backend/employee_act, including subdirectories
    employee_act_dir = os.path.join(os.path.dirname(__file__), 'backend', 'employee_act')
    return send_from_directory(employee_act_dir, filename)

@app.route('/api/postprocess_progress/<recording_id>')
def api_postprocess_progress(recording_id):
    progress = get_postprocess_progress(recording_id)
    return jsonify(progress)

# Global monitoring state
monitoring_active = False
monitoring_thread = None
monitoring_lock = threading.Lock()

# Background monitoring function
def background_employee_monitoring():
    global monitoring_active
    while monitoring_active:
        try:
            # Call your capture logic (simulate POST to capture_employee_activity)
            with app.app_context():
                livefeed.capture_employee_activity()
        except Exception as e:
            print(f"Error in background employee monitoring: {e}")
        # Wait 15 seconds between captures
        time.sleep(15)

@app.route('/start_employee_monitoring', methods=['POST'])
def start_employee_monitoring():
    global monitoring_active, monitoring_thread
    with monitoring_lock:
        # Ensure the background video thread is running
        livefeed.start_background_video_thread()
        if not monitoring_active:
            monitoring_active = True
            monitoring_thread = threading.Thread(target=background_employee_monitoring, daemon=True)
            monitoring_thread.start()
    return jsonify({"success": True, "monitoring": True})

@app.route('/stop_employee_monitoring', methods=['POST'])
def stop_employee_monitoring():
    global monitoring_active
    with monitoring_lock:
        monitoring_active = False
    return jsonify({"success": True, "monitoring": False})

@app.route('/employee_monitoring_status')
def employee_monitoring_status():
    return jsonify({"monitoring": monitoring_active})

@app.route('/employee_act_dir')
@login_required
def get_employee_act_dir():
    # List all files and directories in employee_act for browsing
    employee_act_dir = os.path.join(os.path.dirname(__file__), 'backend', 'employee_act')
    entries = []
    for root, dirs, files in os.walk(employee_act_dir):
        rel_root = os.path.relpath(root, employee_act_dir)
        for d in dirs:
            entries.append({
                'type': 'dir',
                'path': os.path.join(rel_root, d).replace('\\', '/').lstrip('./')
            })
        for f in files:
            entries.append({
                'type': 'file',
                'path': os.path.join(rel_root, f).replace('\\', '/').lstrip('./')
            })
    return jsonify({
        'employee_act_dir': employee_act_dir,
        'entries': entries
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
