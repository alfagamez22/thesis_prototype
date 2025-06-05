from flask import Flask, render_template, Response, jsonify, request, send_from_directory, session, redirect, url_for, flash
import sys
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'har'))
from backend.har import livefeed
from functools import wraps
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
import backend.models as models
from backend.models import db, Employee
from backend.config import Config
from threading import Lock
import threading
import time
# --- WebSocket (Socket.IO) Setup for Real-Time Connected User Count ---
from flask_socketio import SocketIO, emit
from werkzeug.security import check_password_hash, generate_password_hash
import re
import json

app = Flask(__name__)
app.config.from_object(Config)

# Configure file upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'employee_photos')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
def set_postprocess_progress(recording_id, percent, status=None, frame_details=None):
    with postprocess_progress_lock:
        postprocess_progress[recording_id] = {
            'percent': percent,
            'status': status or '',
            'frame_details': frame_details or {}
        }
        print(f"[DEBUG] Progress set for {recording_id}: {percent}% - {status} - Frame details: {frame_details}")  # Debug log

def get_postprocess_progress(recording_id):
    with postprocess_progress_lock:
        return postprocess_progress.get(recording_id, {'percent': 0, 'status': 'Queued', 'frame_details': {}})

# User data file path
USER_DATA_FILE = os.path.join(os.path.dirname(__file__), 'user_data.json')

# Default user data
DEFAULT_USER_DATA = {
    'username': 'username',  # Changed from 'username' to 'admin' for clarity
    'email': 'admin@example.com',
    'password': generate_password_hash('password'),  # Changed to a more secure default password
    'full_name': 'Administrator',
    'phone': '09123456789',
    'dob': datetime(1990, 1, 1).date().isoformat(),
    'gender': 'male',
    'bio': 'System Administrator'
}

def load_user_data():
    """Load user data from JSON file or create with defaults if not exists"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r') as f:
                data = json.load(f)
                # Convert date string back to date object
                data['dob'] = datetime.fromisoformat(data['dob']).date()
                return data
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
            # Save default data to file
            save_user_data(DEFAULT_USER_DATA)
            return DEFAULT_USER_DATA
    except Exception as e:
        print(f"Error loading user data: {e}")
        return DEFAULT_USER_DATA

def save_user_data(data):
    """Save user data to JSON file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(USER_DATA_FILE), exist_ok=True)
        
        # Convert date to ISO format string for JSON serialization
        data_to_save = data.copy()
        data_to_save['dob'] = data['dob'].isoformat()
        
        with open(USER_DATA_FILE, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"User data saved successfully to {USER_DATA_FILE}")
    except Exception as e:
        print(f"Error saving user data: {e}")

# Load user data at startup
USER_DATA = load_user_data()
print(f"Loaded user data: {USER_DATA['username']}")  # Debug print

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Connected User Count API ---
@app.route('/api/connected_users')
def connected_users():
    from backend.har import livefeed
    return livefeed.connected_users_api()

@app.before_request
def track_active_session():
    # Only track for logged-in users
    if session.get('logged_in'):
        if 'sid' not in session:
            session['sid'] = str(uuid.uuid4())
        from backend.har import livefeed
        livefeed.add_active_session(session['sid'])

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        print(f"Login attempt - Username: {username}")  # Debug print
        print(f"Stored username: {USER_DATA['username']}")  # Debug print
        
        # Check against user data
        if username == USER_DATA['username'] and check_password_hash(USER_DATA['password'], password):
            session['logged_in'] = True
            session['username'] = USER_DATA['username']
            print(f"Login successful for user: {username}")  # Debug print
            return redirect(url_for('home'))
        else:
            print(f"Login failed for user: {username}")  # Debug print
            if username != USER_DATA['username']:
                print("Username mismatch")  # Debug print
            else:
                print("Password mismatch")  # Debug print
            flash('Invalid username or password')
    
    return render_template("login.html")

@app.route("/logout")
def logout():
    if 'sid' in session:
        from backend.har import livefeed
        livefeed.remove_active_session(session['sid'])
    session.pop('logged_in', None)
    session.pop('username', None)
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
    return render_template('account.html', user=USER_DATA)

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

@app.route('/video_frame')
def video_frame():
    """Serve a single video frame for Safari compatibility"""
    frame = livefeed.get_latest_frame()
    if frame is not None:
        return Response(frame, mimetype='image/jpeg')
    else:
        # Return a small placeholder image if no frame available
        import base64
        # 1x1 transparent PNG
        placeholder = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQIHWNgAAIAAAUAAY27m/MAAAAASUVORK5CYII=')
        return Response(placeholder, mimetype='image/png')

@app.route('/activity_status')
def activity_status():
    # Returns global activity recognition state
    active = livefeed.get_activity_recognition_state()
    return jsonify({'active': active})

@app.route('/toggle_activity', methods=['POST'])
def toggle_activity():
    # Toggle global activity recognition state
    active = livefeed.toggle_activity_recognition()
    return jsonify({'active': active})

@app.route('/recording_status')
def recording_status():
    # Returns global recording state (both original and segmented)
    status = livefeed.get_recording_status()
    return jsonify(status)

@app.route('/toggle_dual_recording', methods=['POST'])
def toggle_dual_recording():
    # Toggle both original and segmented recording simultaneously
    status = livefeed.get_recording_status()
    if status['original'] or status['segmented']:
        livefeed.stop_dual_recording()
        return jsonify({'recording': False})
    else:
        success = livefeed.start_dual_recording()
        return jsonify({'recording': success})

@app.route('/download_recording/<filename>')
def download_recording(filename):
    return send_from_directory('backend','recordings', filename, as_attachment=True)

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
                         and not f.startswith('raw_segmented_recording_')
                         and not f.startswith('Raw Segmented Recording ')]
        files.extend(original_files)

        # Check for raw segmented files that are being processed (both old and new formats)
        raw_files = [f for f in os.listdir(recordings_dir)
                    if (f.startswith('raw_segmented_recording_') or f.startswith('Raw Segmented Recording ')) 
                    and f.endswith(('.mp4', '.avi'))]
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

# File upload helper functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_photo(file):
    """Save uploaded photo and return the URL path"""
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Return relative URL path
        return f"/static/uploads/employee_photos/{unique_filename}"
    return None

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
    # Handle both JSON and form data (with file uploads)
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Handle form data with file upload
        data = request.form.to_dict()
        
        # Handle photo upload
        photo_url = None
        if 'photoFile' in request.files:
            file = request.files['photoFile']
            if file.filename:
                photo_url = save_uploaded_photo(file)
        
        # Use photoUrl if no file was uploaded
        if not photo_url and data.get('photoUrl'):
            photo_url = data.get('photoUrl')
            
    else:
        # Handle JSON data
        data = request.json
        photo_url = data.get('photoUrl')
    
    # Convert hire_date string to date object
    hire_date = datetime.strptime(data['hireDate'], '%Y-%m-%d').date()
    
    employee = Employee(
        employee_id=data['employeeId'],
        full_name=data['fullName'],
        photo_url=photo_url,
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
    
    # Handle both JSON and form data (with file uploads)
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Handle form data with file upload
        data = request.form.to_dict()
        
        # Handle photo upload
        photo_url = None
        if 'photoFile' in request.files:
            file = request.files['photoFile']
            if file.filename:
                photo_url = save_uploaded_photo(file)
        
        # Use photoUrl if no file was uploaded
        if not photo_url and data.get('photoUrl'):
            photo_url = data.get('photoUrl')
            
        # Update photo URL if we have one
        if photo_url:
            data['photoUrl'] = photo_url
            
    else:
        # Handle JSON data
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

@app.route('/static/uploads/employee_photos/<filename>')
def uploaded_file(filename):
    """Serve uploaded employee photos"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/employee_captures/<path:filename>')
@login_required
def serve_employee_capture(filename):
    """Serve employee capture images from the employee_act directory"""
    try:
        backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
        employee_act_dir = os.path.join(backend_dir, 'employee_act')
        
        # The filename comes in format like "20250601_050442\employee_EMP_001_20250601_050442.jpg"
        # We need to replace backslashes with forward slashes for proper path handling
        filename = filename.replace('\\', '/')
        
        # Split the path to get folder and actual filename
        if '/' in filename:
            folder_name, actual_filename = filename.rsplit('/', 1)
            file_path = os.path.join(employee_act_dir, folder_name, actual_filename)
            serve_dir = os.path.join(employee_act_dir, folder_name)
        else:
            # If no folder, serve directly from employee_act_dir
            file_path = os.path.join(employee_act_dir, filename)
            serve_dir = employee_act_dir
            actual_filename = filename
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Employee capture file not found: {file_path}")
            return "File not found", 404
        
        print(f"Serving employee capture: {filename} from {file_path}")
        
        return send_from_directory(
            serve_dir, 
            actual_filename,
            mimetype='image/jpeg',
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving employee capture {filename}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

@app.route('/api/postprocess_progress/<recording_id>')
def api_postprocess_progress(recording_id):
    progress = get_postprocess_progress(recording_id)
    print(f"[DEBUG] Progress API called for {recording_id}: {progress}")  # Debug log
    return jsonify(progress)

@app.route('/connection_status')
def connection_status():
    from backend.har.livefeed import video_manager
    status = video_manager.get_connection_status()
    return jsonify(status)

@app.route('/activity_logs')
@login_required
def activity_logs():
    """Get activity logs for display in the live feed interface"""
    try:
        import uuid
        
        # Generate or get client session ID
        client_id = session.get('client_id')
        if not client_id:
            client_id = str(uuid.uuid4())
            session['client_id'] = client_id
        
        # Check if this is a request for all logs (initial load)
        load_all = request.args.get('load_all', 'false').lower() == 'true'
        
        if load_all:
            # Return all logs for initial page load
            logs = livefeed.get_all_activity_logs()
        else:
            # Return only new logs since last request
            logs = livefeed.get_recent_activity_logs(client_id)
        
        return jsonify({
            "success": True,
            "logs": logs,
            "client_id": client_id
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "logs": [], 
            "error": str(e)
        })

@app.route('/clear_activity_logs', methods=['POST'])
@login_required
def clear_activity_logs():
    """Clear all activity logs"""
    try:
        success = livefeed.clear_activity_logs()
        return jsonify({
            "success": success,
            "message": "Activity logs cleared successfully" if success else "Failed to clear logs"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/test_activity_log', methods=['POST'])
@login_required
def test_activity_log():
    """Test route to generate a sample activity log"""
    try:
        # Generate a test log entry
        import datetime
        test_detections = {
            'EMP_001': ['walking', 'talking', 'using_phone']
        }
        
        # Call the logging function directly
        livefeed.log_activity_detection_simple(test_detections)
        
        return jsonify({
            "success": True,
            "message": "Test activity log generated"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

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

@app.route('/active_employees')
@login_required
def get_active_employees():
    """Get current count of active employees"""
    try:
        result = livefeed.get_active_employees_count()
        return jsonify(result)
    except Exception as e:
        return jsonify({"active_employees": 0}), 500

@app.route('/api/active_employees')
@login_required
def api_active_employees():
    """API endpoint for active employees count"""
    try:
        result = livefeed.get_active_employees_count()
        return jsonify(result)
    except Exception as e:
        return jsonify({"active_employees": 0}), 500

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

@app.route('/clear_all_employee_images', methods=['DELETE'])
@login_required
def clear_all_employee_images():
    """Clear all images and directories from employee_act directory"""
    try:
        import shutil
        employee_act_dir = os.path.join(os.path.dirname(__file__), 'backend', 'employee_act')
        
        if os.path.exists(employee_act_dir):
            # Remove all contents of the directory but keep the directory itself
            for item in os.listdir(employee_act_dir):
                item_path = os.path.join(employee_act_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                elif os.path.isfile(item_path) and item not in ['ignore.txt', '.gitkeep']:  # Keep ignore files
                    os.remove(item_path)
        
        return jsonify({"success": True, "message": "All employee images and metadata cleared successfully"})
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- SocketIO Setup ---
socketio = SocketIO(app, cors_allowed_origins="*")

# Use a set to track connected Socket.IO session IDs
active_ws_sessions = set()
from threading import Lock
active_ws_sessions_lock = Lock()

def broadcast_user_count():
    with active_ws_sessions_lock:
        count = len(active_ws_sessions)
    socketio.emit('user_count', {'connected_users': count})

@socketio.on('connect')
def handle_connect():
    with active_ws_sessions_lock:
        active_ws_sessions.add(request.sid)
    broadcast_user_count()

@socketio.on('disconnect')
def handle_disconnect():
    with active_ws_sessions_lock:
        active_ws_sessions.discard(request.sid)
    broadcast_user_count()

@app.route('/list_all_employee_captures')
@login_required
def list_all_employee_captures():
    """Get all employee captures with metadata from stored files"""
    try:
        result = livefeed.get_all_employee_captures_with_metadata()
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    # Update profile information
    USER_DATA['full_name'] = request.form.get('fullName')
    USER_DATA['phone'] = request.form.get('phone')
    USER_DATA['dob'] = datetime.strptime(request.form.get('dob'), '%Y-%m-%d').date()
    USER_DATA['gender'] = request.form.get('gender')
    USER_DATA['bio'] = request.form.get('bio')
    
    # Validate phone number
    if not re.match(r'^09\d{9}$', USER_DATA['phone']):
        return jsonify({'error': 'Invalid phone number format'}), 400
    
    # Save changes to file
    save_user_data(USER_DATA)
    return jsonify({'message': 'Profile updated successfully'}), 200

@app.route('/update_password', methods=['POST'])
@login_required
def update_password():
    current_password = request.form.get('currentPassword')
    new_password = request.form.get('newPassword')
    confirm_password = request.form.get('confirmPassword')
    
    # Verify current password
    if not check_password_hash(USER_DATA['password'], current_password):
        return jsonify({'error': 'Current password is incorrect'}), 400
    
    # Verify new passwords match
    if new_password != confirm_password:
        return jsonify({'error': 'New passwords do not match'}), 400
    
    # Update password
    USER_DATA['password'] = generate_password_hash(new_password)
    
    # Save changes to file
    save_user_data(USER_DATA)
    return jsonify({'message': 'Password updated successfully'}), 200

@app.route('/update_username', methods=['POST'])
@login_required
def update_username():
    new_username = request.form.get('newUsername')
    current_password = request.form.get('usernamePassword')
    
    # Verify current password
    if not check_password_hash(USER_DATA['password'], current_password):
        return jsonify({'error': 'Current password is incorrect'}), 400
    
    # Update username
    USER_DATA['username'] = new_username
    session['username'] = new_username
    
    # Save changes to file
    save_user_data(USER_DATA)
    return jsonify({'message': 'Username updated successfully'}), 200

@app.route('/update_email', methods=['POST'])
@login_required
def update_email():
    new_email = request.form.get('newEmail')
    current_password = request.form.get('emailPassword')
    
    # Verify current password
    if not check_password_hash(USER_DATA['password'], current_password):
        return jsonify({'error': 'Current password is incorrect'}), 400
    
    # Update email
    USER_DATA['email'] = new_email
    
    # Save changes to file
    save_user_data(USER_DATA)
    return jsonify({'message': 'Email updated successfully'}), 200

# Admin credentials (hardcoded)
ADMIN_CREDENTIALS = {
    'username': 'admin',
    'password': generate_password_hash('password')
}

# Store submitted tickets
PASSWORD_RESET_TICKETS = []

@app.route("/forgot_password", methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Store the ticket
        ticket = {
            'email': email,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        PASSWORD_RESET_TICKETS.append(ticket)
        flash('Ticket submitted successfully. Please wait for admin to contact you.')
        return redirect(url_for('login'))
    return render_template("forgot_password.html")

@app.route("/admin/login", methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == ADMIN_CREDENTIALS['username'] and check_password_hash(ADMIN_CREDENTIALS['password'], password):
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials')
    
    return render_template("admin_login.html")

@app.route("/admin/dashboard")
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    
    # Get all pending tickets
    pending_tickets = [t for t in PASSWORD_RESET_TICKETS if t['status'] == 'pending']
    
    # Load current user data
    current_user_data = load_user_data()
    
    return render_template("admin_dashboard.html", 
                         tickets=pending_tickets,
                         user_data=current_user_data)

@app.route("/admin/update_user_credentials", methods=['POST'])
def update_user_credentials():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    new_username = request.form.get('new_username')
    new_password = request.form.get('new_password')
    
    if not new_username or not new_password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    try:
        # Load current user data
        current_user_data = load_user_data()
        
        # Update user data
        current_user_data['username'] = new_username
        current_user_data['password'] = generate_password_hash(new_password)
        
        # Save changes to file
        save_user_data(current_user_data)
        
        # Update global USER_DATA
        global USER_DATA
        USER_DATA = current_user_data
        
        # Update session if the current user is logged in
        if session.get('username') == current_user_data['username']:
            session['username'] = new_username
        
        print(f"User credentials updated - New username: {new_username}")  # Debug print
        return jsonify({'message': 'User credentials updated successfully'})
    except Exception as e:
        print(f"Error updating user credentials: {e}")  # Debug print
        return jsonify({'error': f'Failed to update credentials: {str(e)}'}), 500

@app.route("/admin/resolve_ticket", methods=['POST'])
def resolve_ticket():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    ticket_index = request.form.get('ticket_index')
    if ticket_index is not None:
        try:
            PASSWORD_RESET_TICKETS[int(ticket_index)]['status'] = 'resolved'
            return jsonify({'message': 'Ticket resolved successfully'})
        except (IndexError, ValueError):
            return jsonify({'error': 'Invalid ticket index'}), 400
    
    return jsonify({'error': 'Ticket index required'}), 400

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

if __name__ == "__main__":
    # Ensure user data is properly initialized
    if not os.path.exists(USER_DATA_FILE):
        print("Initializing user data file with default credentials:")
        print(f"Username: {DEFAULT_USER_DATA['username']}")
        print(f"Password: password")  # Don't print the hashed password
        save_user_data(DEFAULT_USER_DATA)
    
    # Run the app with SocketIO
    # print("Starting server on http://localhost:5000")
    # app.run(host="127.0.0.1", port=5000, debug=True)  # Changed from 0.0.0.0 to 127.0.0.1
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)