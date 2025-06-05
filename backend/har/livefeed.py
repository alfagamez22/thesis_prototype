import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import time
import threading
import datetime
import json
import collections
import sys
import traceback
from typing import List, Dict, Tuple, Optional
from PIL import Image
from torchvision.transforms import v2
import av  # PyAV for robust RTSP streaming
import queue

# ----------------------
# Configuration
# ----------------------
class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Human detection with action recognition")
        parser.add_argument("--REC", action='store_true', help="record stream")
        parser.add_argument("-weights", type=str, 
                           default=os.path.join(os.path.dirname(__file__), "weights", "avak_b16.pt"),
                           help="path to action recognition pretrained weights")
        # D:\JULIAN THE MAN\COLLEGE\FEU\THESIS\final\flask\backend\weights\avak_b16.pt
        # default=r"C:\\Users\\buanh\\Documents\\VSCODE\\thesis\\livefeed2\\webapp\\flask-template\\backend\\weights\\avak_b16.pt",
        parser.add_argument("-rate", type=int, default=2, help="sampling rate")
        parser.add_argument("-act_thresh", type=float, default=0.23, help="action confidence threshold")
        parser.add_argument("-det_thresh", type=float, default=0.6, help="person detection confidence threshold")
        parser.add_argument("-nms_thresh", type=float, default=0.75, help="NMS threshold for person detection")
        parser.add_argument("-act", type=lambda s: s.split(","), help="Comma-separated list of actions")
        parser.add_argument("-color", type=str, default='green', help="color to plot predictions")
        parser.add_argument("-font", type=float, default=0.7, help="font size")
        parser.add_argument("-line", type=int, default=2, help="line thickness")
        parser.add_argument("-delay", type=int, default=30, help="frame delay amount")
        parser.add_argument("-fps", type=int, default=10, help="target frames per second")
        parser.add_argument("-det_freq", type=int, default=6, help="detection frequency (every N frames)")
        #parser.add_argument("-F", type=str, default=os.path.join(os.path.dirname(__file__), "assets", "COFFEESHOP_1.mp4"), help="file path for video input")
        parser.add_argument("-F", type=str, default=os.path.join(os.path.dirname(__file__), "assets", "rtsp.mp4"), help="file path for video input")
        # parser.add_argument("-F", type=str, default="<REPLACE> rtsp://username:password@tunnelip:8554/stream2?tcp <REPLACE>", help="file path for video input") #rtsp://username:password@tunnelip:8554/stream2?tcp
     #   parser.add_argument("-F", type=str, default="rtsp://harveybuan123:harveybuan1234@100.107.152.111:8554/my_camera?tcp", help="file path for video input") #for rtsp stream 192.168.68.128:554 #rtsp://harveybuan123:harveybuan1234@100.107.152.111:8554/stream2?tcp
    #    parser.add_argument("-F", type=str, default="rtsp://harveybuan123:harveybuan1234@100.107.152.111:8554/my_camera?tcp", help="file path for video input") #for rtsp stream 192.168.68.128:554 #rtsp://harveybuan123:harveybuan1234@100.107.152.111:8554/stream2?tcp
        parser.add_argument("-roi", type=str, default=None, help="Path to ROI coordinates JSON")
        # Optimize: Only parse args if run as main, otherwise use defaults (empty list)
        self.args = parser.parse_args([])

        # Load action classes
        if self.args.act is None:
            with open(os.path.join(os.path.dirname(__file__), 'ava_classes.json'), 'r') as f:
                self.args.act = json.load(f)
        else:
            self.args.act = [s.replace('_', ' ') for s in self.args.act]

        # Color mapping
        self.COLORS = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }
        self.color = self.COLORS.get(self.args.color, (0, 255, 0))
        self.font = self.args.font
        self.thickness = self.args.line        # Fixed display dimensions
        self.display_width, self.display_height = 1280, 720

        # Fixed ROI polygon (hardcoded from user selection)
        self.roi_polygon = [
            (1194, 358),
            (585, 355),
            (389, 232),
            (236, 115),
            (12, 20)
        ]
        
        # Keep backward compatibility with line-based ROI
        self.roi_line_p1 = (1163, 425)
        self.roi_line_p2 = (97, 121)
        
        #for recorded coffee shop 
        #self.roi_line_p1 = (7, 299)
        #self.roi_line_p2 = (1253, 675)

        # Set video path
        # if self.args.F and not os.path.isabs(self.args.F):
        #     self.args.F = os.path.join(os.path.dirname(__file__), self.args.F)
        if self.args.F and not (self.args.F.startswith("rtsp://") or self.args.F.startswith("http://") or os.path.isabs(self.args.F)):
            self.args.F = os.path.join(os.path.dirname(__file__), self.args.F)

# ----------------------
# Model Setup
# ----------------------
class ModelManager:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Configure CUDA for performance
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Set up RT-DETR detector
        self.setup_detector()
        
        # Set up action recognition model
        self.setup_action_model()
        
        # Image processing setup
        self.imgsize = (240, 320)
        self.tfs = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # Action postprocessing
        self.postprocess = PostProcessActions()
        
        # Toggle states
        self.activity_recognition_on = True

    def setup_detector(self):
        """Initialize the RT-DETR person detector"""
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
        print("Loading RT-DETR model...")
        model_name = "PekingU/rtdetr_v2_r18vd"
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_name)
        self.rt_detr_model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
        
        # Use half precision if supported hardware is available
        self.use_half_precision = (self.device.type == "cuda" and 
                                  torch.cuda.is_available() and 
                                  torch.cuda.get_device_capability(0)[0] >= 7)
        if self.use_half_precision:
            self.rt_detr_model = self.rt_detr_model.half()
            print("Using half precision (FP16) for faster inference")
            
        self.rt_detr_model = self.rt_detr_model.to(self.device).eval()

    def setup_action_model(self):
        """Initialize the action recognition model"""
        from .hb import get_hb
        print("Loading Action Recognition model...")
        self.action_model = get_hb(size='b', pretrain=None, det_token_num=20, 
                              text_lora=True, num_frames=9)['hb']
        self.action_model.load_state_dict(
            torch.load(self.config.args.weights, map_location='cpu'), 
            strict=False
        )
        self.action_model.to(self.device).eval()
        
        # Process action captions
        self.captions = self.config.args.act
        self.text_embeds = F.normalize(self.action_model.encode_text(self.captions), dim=-1)
    
    def detect_persons(self, frame):
        """Detect persons in a frame using RT-DETR model"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if self.use_half_precision:
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.rt_detr_model(**inputs)
            
        target_sizes = torch.tensor([(pil_image.height, pil_image.width)]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=self.config.args.det_thresh
        )[0]
        
        results = {k: v.cpu() for k, v in results.items()}
        
        # Filter for persons only
        person_boxes_xyxy, person_scores = [], []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.rt_detr_model.config.id2label[label.item()].lower() == "person":
                person_scores.append(score.item())
                person_boxes_xyxy.append([int(i) for i in box.tolist()])
                
        # Apply NMS
        person_detections = []
        if person_boxes_xyxy:
            indices = cv2.dnn.NMSBoxes(
                person_boxes_xyxy, 
                person_scores, 
                self.config.args.det_thresh, 
                self.config.args.nms_thresh
            )
            for idx in indices:
                idx = idx[0] if isinstance(idx, np.ndarray) else idx
                person_detections.append({
                    "score": person_scores[idx], 
                    "box": person_boxes_xyxy[idx]
                })
                
        return person_detections
    
    def recognize_actions(self, buffer, persons_in_roi):
        """Recognize actions for people in ROI using the action model"""
        if not self.activity_recognition_on or len(buffer) < 9:
            return {}
        
        try:
            frames_to_use = buffer[-9:]  # Use last 9 frames
            clip_torch = torch.from_numpy(np.array(frames_to_use)).to(self.device) / 255
            clip_torch = self.tfs(clip_torch)
            
            person_actions = {}
            for i, person in enumerate(persons_in_roi):
                with torch.no_grad():
                    features = self.action_model.encode_vision(clip_torch.unsqueeze(0))
                    action_probs = F.normalize(features['pred_logits'], dim=-1) @ self.text_embeds.T
                    actions = self.postprocess(action_probs, threshold=self.config.args.act_thresh)
                    person_actions[i] = actions
            return person_actions
            
        except Exception as e:
            print(f"Error in action recognition: {e}")
            return {}
            
    def toggle_activity_recognition(self):
        """Toggle the activity recognition feature on/off"""
        self.activity_recognition_on = not self.activity_recognition_on
        print(f"Human activity recognition toggled to: {self.activity_recognition_on}")
        return self.activity_recognition_on
        
    def get_activity_recognition_state(self):
        """Get current state of activity recognition"""
        return self.activity_recognition_on

# ----------------------
# Utilities
# ----------------------
class PostProcessActions:
    """Process action recognition results"""
    def __call__(self, action_probs, threshold=0.25):
        # Keep everything on GPU until numpy conversion is needed
        if len(action_probs.shape) == 3:
            probs = action_probs[0, 0]
        elif len(action_probs.shape) == 2:
            probs = action_probs[0]
        else:
            probs = action_probs
            
        mask = probs > threshold
        indices = torch.nonzero(mask).flatten()
        actions = [(int(i), float(probs[i].item())) for i in indices]
        return sorted(actions, key=lambda x: x[1], reverse=True)

def is_on_active_side(pt, line_p1, line_p2):
    """Check if a point is on the 'active' side of the ROI line"""
    x, y = pt
    x1, y1 = line_p1
    x2, y2 = line_p2
    # Vector from p1 to p2
    dx, dy = x2 - x1, y2 - y1
    # Vector from p1 to pt
    dxp, dyp = x - x1, y - y1
    # Cross product
    cross = dx * dyp - dy * dxp
    return cross > 0  # Positive cross product defines the active side

def is_point_in_polygon(pt, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = pt
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

# ----------------------
# Video and Recording Management
# ----------------------
class VideoManager:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        
        # Video buffers
        self.buffer = []         # Action recognition buffer (processed frames)
        # ---
        # Delay buffer: This buffer is used to introduce a fixed delay in the video stream output.
        # Frames are appended to this buffer, and only after the buffer has enough frames (delay + margin),
        # the oldest frame is popped and processed. This creates a fixed delay (in frames) between capture and display,
        # which is useful for synchronizing streams, smoothing network jitter, or meeting application requirements.
        # The delay amount is set by self.config.args.delay (default: 30 frames).
        # ---
        self.delay_buffer = collections.deque()  # (frame, timestamp)
        self.delay_seconds = 2.0  # Set your desired delay here
        
        # State variables
        self.fps_history = []
        self.last_person_detections = []
        self.frame_count = 0
        self.roi_id = 1
        
        # Threading
        self.latest_frame = None
        self.video_thread_started = False
        self.video_thread_lock = threading.Lock()
        
        # Recording setup - support for two types
        self.original_writer = None
        self.segmented_writer = None
        self.original_recording_active = False
        self.segmented_recording_active = False
        if self.config.args.REC:
            self.setup_recording()
            
        self.frame_queue = queue.Queue(maxsize=30)  # For decoupled RTSP reading
        self.rtsp_reader_thread = None

    def setup_recording(self, recording_type="original"):
        """Setup video recording for either original or segmented recording"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d __ %I-%M-%S_%p")
        recordings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "recordings")
        if not os.path.exists(recordings_dir):
            os.makedirs(recordings_dir)

        if recording_type == "segmented":
            output_path = os.path.join(recordings_dir, f"Raw Segmented Recording {timestamp}.mp4")
        else:
            output_path = os.path.join(recordings_dir, f"Original Recording {timestamp}.mp4")

        # Chrome-compatible H.264 codecs (baseline profile first for best compatibility)
        mp4_codecs = ['avc1', 'X264', 'H264', 'mp4v']
        success = False
        writer = None

        for codec in mp4_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    max(5, min(self.config.args.fps, 30)),
                    (self.config.display_width, self.config.display_height)
                )
                if writer.isOpened():
                    print(f"{recording_type.capitalize()} recording started with {codec} codec: {output_path}")
                    if recording_type == "segmented":
                        self.segmented_writer = writer
                        self.current_segmented_recording_path = output_path
                        self.segmented_recording_active = True
                    else:
                        self.original_writer = writer
                        self.current_original_recording_path = output_path
                        self.original_recording_active = True
                    success = True
                    break
                else:
                    if writer:
                        writer.release()
                    print(f"Failed to initialize {recording_type} recording with {codec} codec, trying next...")
            except Exception as e:
                print(f"Error with {codec} codec for {recording_type} recording: {e}")
                if writer:
                    writer.release()
                continue
        if not success:
            print(f"ERROR: Could not initialize {recording_type} video writer with any MP4 codec!")
            if recording_type == "segmented":
                self.segmented_writer = None
            else:
                self.original_writer = None

    def stop_recording(self, recording_type="original"):
        """Stop video recording for the specified type"""
        if recording_type == "segmented":
            if self.segmented_writer is not None:
                self.segmented_writer.release()
                self.segmented_writer = None
                self.segmented_recording_active = False
                if hasattr(self, 'current_segmented_recording_path'):
                    print(f"Segmented recording stopped and saved: {self.current_segmented_recording_path}")
                    # Verify the file was created and has content
                    if os.path.exists(self.current_segmented_recording_path):
                        file_size = os.path.getsize(self.current_segmented_recording_path)
                        print(f"Segmented recording file size: {file_size} bytes")
                        
                        # Start post-processing in background thread
                        threading.Thread(
                            target=self._post_process_segmented_recording,
                            args=(self.current_segmented_recording_path,),
                            daemon=True
                        ).start()
                    else:
                        print("ERROR: Segmented recording file was not created!")
                else:
                    print("Segmented recording stopped")
        else:
            if self.original_writer is not None:
                self.original_writer.release()
                self.original_writer = None
                self.original_recording_active = False
                if hasattr(self, 'current_original_recording_path'):
                    print(f"Original recording stopped and saved: {self.current_original_recording_path}")
                    # Verify the file was created and has content
                    if os.path.exists(self.current_original_recording_path):
                        file_size = os.path.getsize(self.current_original_recording_path)
                        print(f"Original recording file size: {file_size} bytes")
                    else:
                        print("ERROR: Original recording file was not created!")
                else:
                    print("Original recording stopped")
    
    def _post_process_segmented_recording(self, raw_recording_path):
        """Post-process the raw segmented recording using auto.py"""
        try:
            print(f"Starting post-processing of: {raw_recording_path}")
            
            # Create segmented recordings directory
            recordings_dir = os.path.dirname(raw_recording_path)
            segmented_dir = os.path.join(recordings_dir, "segmented_recordings")
            if not os.path.exists(segmented_dir):
                os.makedirs(segmented_dir)
            
            # Generate output path for processed video
            filename = os.path.basename(raw_recording_path)
            # Handle both old and new raw filename formats
            if filename.startswith("raw_segmented_recording_"):
                processed_filename = filename.replace("raw_segmented_recording_", "Segmented Recording ")
            elif filename.startswith("Raw Segmented Recording "):
                processed_filename = filename.replace("Raw ", "")
            else:
                processed_filename = filename
            processed_path = os.path.join(segmented_dir, processed_filename)

            # Import and run auto.py processing
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'postprocess'))
            from auto import auto_annotate_video_with_models
            # Import the progress setter from app.py
            from app import set_postprocess_progress

            # Define a callback that updates progress in the Flask app
            def progress_callback(recording_id, percent, frame_details=None):
                status = 'Processing'
                if frame_details:
                    status = frame_details.get('current_status', 'Processing')
                set_postprocess_progress(recording_id, int(percent * 100), status=status, frame_details=frame_details)

            # Run the auto annotation with progress reporting
            auto_annotate_video_with_models(
                video_path=raw_recording_path,
                output_video_path=processed_path,
                det_model="PekingU/rtdetr_v2_r18vd",
                sam_model_path="mobile_sam.pt",  # Changed from sam2.1_t.pt to mobile_sam.pt
                device="cuda" if torch.cuda.is_available() else "cpu",
                conf=0.55,
                visualize=True,
                use_sam=True,  # Will fallback to False if SAM model fails to load
                iou_threshold=0.3,
                progress_callback=progress_callback,
                recording_id=filename
            )
            
            # Remove the raw recording file after successful processing
            if os.path.exists(processed_path):
                os.remove(raw_recording_path)
                set_postprocess_progress(filename, 100, status='Complete')
                print(f"Post-processing complete! Processed video saved at: {processed_path}")
                print(f"Raw recording file removed: {raw_recording_path}")
            else:
                set_postprocess_progress(filename, 100, status='Failed')
                print(f"ERROR: Post-processing failed, processed file not found: {processed_path}")
                
        except Exception as e:
            from app import set_postprocess_progress
            set_postprocess_progress(os.path.basename(raw_recording_path), 100, status=f'Error: {e}')
            print(f"ERROR in post-processing: {e}")
            traceback.print_exc()
            
    def process_frame(self, frame):
        """Process a single frame and return the annotated output"""
        self.frame_count += 1
        
        # Flip and resize the frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.config.display_width, self.config.display_height))
        
        # Store current frame for employee capture
        self.current_frame = frame.copy()
        
        # Add to delay buffer
        self.delay_buffer.append(frame.copy())
        
        # Process only if we have enough frames in the delay buffer
        if len(self.delay_buffer) <= self.config.args.delay + 20:
            return frame  # Return original frame if not enough delay buffer
            
        # Get delayed frame and process
        delayed_frame = self.delay_buffer.popleft()
        
        # Person detection (only run every det_freq frames for better performance)
        person_detections = []
        if self.frame_count % self.config.args.det_freq == 0:
            person_detections = self.model_manager.detect_persons(delayed_frame)
            self.last_person_detections = person_detections
        else:
            person_detections = self.last_person_detections
            
        # Store current detections for employee capture
        # Convert detections to simple format for employee capture
        self.current_detections = []
        for person in person_detections:
            box, score = person["box"], person["score"]
            # Store as [x1, y1, x2, y2, confidence]
            self.current_detections.append([box[0], box[1], box[2], box[3], score])
            
        # Process frame for action recognition
        delayed_frame_resized = cv2.resize(
            delayed_frame, 
            (self.model_manager.imgsize[1], self.model_manager.imgsize[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        color_image = delayed_frame_resized.transpose(2, 0, 1)
        self.buffer.append(color_image)
        
        # Keep buffer size managed
        if len(self.buffer) > 2 * 9:  # 9 is required_frames
            self.buffer.pop(0)
            
        # Find persons in ROI and track their original indices
        out_frame = delayed_frame.copy()
        persons_in_roi = []
        roi_to_original_mapping = {}  # Maps ROI index to original detection index
        
        for i, p in enumerate(person_detections):
            box = p["box"]
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            if is_on_active_side((cx, cy), self.config.roi_line_p1, self.config.roi_line_p2):
                roi_index = len(persons_in_roi)
                persons_in_roi.append(p)
                roi_to_original_mapping[roi_index] = i
                
        # Action recognition
        person_actions = {}
        if self.model_manager.activity_recognition_on and len(self.buffer) >= 9:
            person_actions = self.model_manager.recognize_actions(self.buffer, persons_in_roi)
            
        # Store current activities for employee capture using original detection indices
        self.current_activities = {}
        for roi_idx, actions in person_actions.items():
            original_idx = roi_to_original_mapping.get(roi_idx)
            if original_idx is not None:
                # Convert actions to readable format
                action_names = []
                for action_idx, confidence in actions:
                    if action_idx < len(self.model_manager.captions):
                        action_names.append(self.model_manager.captions[action_idx])
                self.current_activities[original_idx] = action_names
            
        # Draw ROI line
        cv2.line(out_frame, self.config.roi_line_p1, self.config.roi_line_p2, (0, 255, 255), 2)
        
        # Draw bounding boxes and annotations
        person_count = 0
        self.roi_id = 1
        
        for i, person in enumerate(person_detections):
            box, score = person["box"], person["score"]
            cv2.rectangle(
                out_frame, 
                (box[0], box[1]), 
                (box[2], box[3]), 
                self.config.color, 
                self.config.thickness
            )
            
            label_text = f"Person: {score:.2f}"
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            
            if is_on_active_side((cx, cy), self.config.roi_line_p1, self.config.roi_line_p2):
                label_text += f" | ROI: employeeID:[{self.roi_id}]"
                self.roi_id += 1
                person_count += 1
                
            cv2.putText(
                out_frame, 
                label_text, 
                (box[0], box[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.config.font, 
                self.config.color, 
                self.config.thickness
            )
            
            # Display actions for this person
            if (self.model_manager.activity_recognition_on and 
                i in person_actions and 
                person_actions[i]):
                y_offset = 25
                for action_idx, confidence in person_actions[i]:
                    action_text = f"{self.model_manager.captions[action_idx]}: {confidence:.2f}"
                    cv2.putText(
                        out_frame, 
                        action_text, 
                        (box[0], box[1] + y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        self.config.font, 
                        self.config.color, 
                        self.config.thickness
                    )
                    y_offset += 25
                    
        # Draw stats
        cv2.putText(
            out_frame, 
            f"Total Persons: {len(person_detections)}", 
            (self.config.display_width - 320, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (255, 0, 0), 
            3
        )
        
        cv2.putText(
            out_frame, 
            f"Persons in ROI: {person_count}", 
            (10, self.config.display_height - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 255), 
            2
        )
        
        # Calculate and display FPS
        current_time = time.time()
        if self.frame_count % 10 == 0:
            fps = 10 / (current_time - self.start_time) if hasattr(self, 'start_time') else 10
            self.fps_history.append(fps)
            if len(self.fps_history) > 10:
                self.fps_history.pop(0)
            self.start_time = current_time
            
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        cv2.putText(
            out_frame, 
            f"FPS: {avg_fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            self.config.font, 
            (0, 0, 255), 
            self.config.thickness
        )
        
        # Write frame to video if recording (optimized for dual recording)
        if self.original_writer is not None and self.original_recording_active:
            self.original_writer.write(out_frame)
        if self.segmented_writer is not None and self.segmented_recording_active:
            self.segmented_writer.write(out_frame)
        if self.frame_count % 30 == 0:
            original_status = "ON" if self.original_recording_active else "OFF"
            segmented_status = "ON (will post-process)" if self.segmented_recording_active else "OFF"
            print(f"Recording status - Original: {original_status}, Segmented: {segmented_status}, Frame: {self.frame_count}")
        if self.frame_count % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log activity detection for persons in ROI with detected actions
        if (self.model_manager.activity_recognition_on and 
            person_actions and 
            person_count > 0):
            # Log activities using the existing function signature
            log_activity_detection(person_detections, person_actions, roi_to_original_mapping)
            
        return out_frame
        
    def start_background_video_thread(self):
        """Start video processing in a background thread"""
        if not self.video_thread_started:
            self.start_time = time.time()
            t = threading.Thread(target=self._video_background_worker, daemon=True)
            t.start()
            self.video_thread_started = True
        
    def _start_rtsp_reader_thread(self, video_source):
        """Start a dedicated thread to read RTSP frames as fast as possible and put them in a queue (leaky buffer)."""
        def rtsp_reader():
            try:
                container = av.open(video_source, options={'rtsp_transport': 'tcp', 'max_delay': '5000000'})
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        img = frame.to_ndarray(format='bgr24')
                        while True:
                            try:
                                self.frame_queue.put_nowait(img)
                                break
                            except queue.Full:
                                # Leak downstream: remove oldest frame to make space for the newest
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    break
            except Exception as e:
                print(f"[RTSP Reader] Error: {e}")
        if self.rtsp_reader_thread is None or not self.rtsp_reader_thread.is_alive():
            self.frame_queue.queue.clear()
            self.rtsp_reader_thread = threading.Thread(target=rtsp_reader, daemon=True)
            self.rtsp_reader_thread.start()

    def _video_background_worker(self):
        """Worker function for background video processing with auto-reconnect on RTSP drop and connection speed status"""
        reconnect_interval = 0.5  # 500ms
        video_source = self.config.args.F if self.config.args.F else 0
        last_status = None

        while True:
            use_pyav = isinstance(video_source, str) and video_source.startswith("rtsp://")
            frame_generator = None

            if use_pyav:
                print("[LiveFeed] Using PyAV (threaded) for RTSP stream.")
                self._start_rtsp_reader_thread(video_source)
                def frame_gen():
                    while True:
                        try:
                            frame = self.frame_queue.get(timeout=2)
                            yield frame
                        except queue.Empty:
                            print("[LiveFeed] Frame queue empty, RTSP may be stalled. Reconnecting...")
                            break
                frame_generator = frame_gen()
            else:
                print("[LiveFeed] Using OpenCV for local/camera stream.")
                cap = cv2.VideoCapture(video_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
                if not cap.isOpened():
                    self.connection_status = {
                        'connected': False,
                        'speed': 0,
                        'last_success': None,
                        'error': 'Cannot open video source'
                    }
                    time.sleep(reconnect_interval)
                    continue
                frame_generator = (cap.read()[1] for _ in iter(int, 1))

            self.connection_status = {
                'connected': True,
                'speed': 0,
                'last_success': time.time(),
                'error': None
            }
            print(f"[LiveFeed] Video source opened: {video_source}")

            try:
                for frame in frame_generator:
                    frame_start_time = time.time()
                    if frame is None:
                        print("[LiveFeed] Frame read failed, will reconnect.")
                        self.connection_status = {
                            'connected': False,
                            'speed': 0,
                            'last_success': self.connection_status.get('last_success'),
                            'error': 'Frame read failed'
                        }
                        break
                    processed_frame = self.process_frame(frame)
                    if processed_frame is not None:
                        try:
                            ret, jpeg = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            if ret:
                                with self.video_thread_lock:
                                    self.latest_frame = jpeg.tobytes()
                        except Exception as e:
                            print(f"Error encoding frame: {e}")
                    # Limit playback to target FPS
                    target_frame_time = 1.0 / self.config.args.fps
                    delay = max(0.001, target_frame_time - (time.time() - frame_start_time))
                    time.sleep(delay)
            except Exception as e:
                print(f"[LiveFeed] Error occurred: {e}")
                import traceback
                traceback.print_exc()
                self.connection_status = {
                    'connected': False,
                    'speed': 0,
                    'last_success': self.connection_status.get('last_success'),
                    'error': str(e)
                }
            finally:
                if not use_pyav:
                    cap.release()
                print("[LiveFeed] Video thread cleanup complete. Will attempt to reconnect if needed.")
            time.sleep(reconnect_interval)

    def get_connection_status(self):
        """Return the current connection status and latency (ms) for UI display"""
        # Calculate latency in ms if possible
        status = getattr(self, 'connection_status', {'connected': False, 'speed': 0, 'last_success': None, 'error': 'Not started'})
        # If speed is FPS, convert to ms latency if possible
        if status.get('connected') and status.get('speed', 0) > 0:
            # Use FPS to estimate latency: latency (ms) = 1000 / FPS
            status = status.copy()
            status['latency_ms'] = round(1000.0 / status['speed'], 1)
        else:
            status = status.copy()
            status['latency_ms'] = None
        return status
            
    def get_latest_frame(self):
        """Get the latest processed frame"""
        with self.video_thread_lock:
            return self.latest_frame
            
    def frame_generator(self):
        """Generator that yields processed frames as JPEG for Flask streaming"""
        cap = cv2.VideoCapture(self.config.args.F if self.config.args.F else 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 30)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.config.args.F if self.config.args.F else 0}")
            
        self.start_time = time.time()
        
        try:
            while True:
                frame_start_time = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                    
                processed_frame = self.process_frame(frame)
                if processed_frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', processed_frame)
                    if ret:
                        yield jpeg.tobytes()
                
                # Limit playback to target FPS
                elapsed = time.time() - frame_start_time
                delay = max(1, int((1.0 / self.config.args.fps - elapsed) * 1000))
                time.sleep(delay / 1000.0)
                
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            print("Frame generator cleanup complete.")
    
    def _pyav_frame_reader(self, video_source):
        """Robust RTSP frame reader using PyAV (FFmpeg)"""
        try:
            container = av.open(video_source, options={'rtsp_transport': 'tcp', 'max_delay': '5000000'})
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    yield img
        except Exception as e:
            print(f"[PyAV] Error reading RTSP stream: {e}")
            return

# ----------------------
# Employee Activity Monitoring
# ----------------------
# Employee activity storage
employee_captures = []
employee_act_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'employee_act')

# Create employee_act directory if it doesn't exist
if not os.path.exists(employee_act_dir):
    os.makedirs(employee_act_dir)

# Activity logs storage - persistent across sessions
activity_logs = []
activity_logs_lock = threading.Lock()
# Track last delivered log index for each client session
client_last_log_index = {}
client_sessions_lock = threading.Lock()

def log_activity_detection(person_detections, person_actions, roi_to_original_mapping):
    """Log activity detections for the live feed logs"""
    try:
        if not person_actions:
            return
        
        timestamp = datetime.datetime.now()
        
        with activity_logs_lock:
            # Create log entries for each person with detected activities
            for roi_idx, actions in person_actions.items():
                if not actions:  # Skip if no actions detected
                    continue
                
                original_idx = roi_to_original_mapping.get(roi_idx)
                if original_idx is None:
                    continue
                
                # Generate employee ID based on detection order
                employee_id = f"EMP_{roi_idx + 1:03d}"
                
                # Extract action names - we'll need to access this through a global or pass it as parameter
                action_names = []
                for action_idx, confidence in actions:
                    # Use a global captions list or import from a shared module
                    try:
                        # Try to access captions through the global video_manager if available
                        if 'video_manager' in globals() and hasattr(video_manager, 'model_manager'):
                            captions = video_manager.model_manager.captions
                        else:
                            # Fallback: load captions from the JSON file
                            with open(os.path.join(os.path.dirname(__file__), 'ava_classes.json'), 'r') as f:
                                captions = json.load(f)
                        
                        if action_idx < len(captions):
                            action_names.append(captions[action_idx])
                    except Exception as e:
                        action_names.append(f"Action_{action_idx}")
                
                if action_names:  # Only log if there are valid action names
                    log_entry = {
                        "timestamp": timestamp.isoformat(),
                        "employee_id": employee_id,
                        "actions": action_names,
                        "confidence_scores": [conf for _, conf in actions]
                    }
                    
                    activity_logs.append(log_entry)
            
            # No longer truncate activity_logs, allow it to grow unbounded
            # if len(activity_logs) > 100:
            #     activity_logs[:] = activity_logs[-100:]
                
    except Exception as e:
        print(f"Error logging activity detection: {e}")

def get_recent_activity_logs(client_id=None):
    """Get activity logs for a specific client, returning only new logs since last request"""
    try:
        with activity_logs_lock:
            if not client_id:
                # Return all logs if no client ID provided (for compatibility)
                return activity_logs[:]
            
            with client_sessions_lock:
                # Get the last index this client received
                last_index = client_last_log_index.get(client_id, 0)
                
                # Get new logs since last request
                if last_index < len(activity_logs):
                    new_logs = activity_logs[last_index:]
                    # Update client's last index
                    client_last_log_index[client_id] = len(activity_logs)
                    return new_logs
                else:
                    # No new logs
                    return []
                    
    except Exception as e:
        print(f"Error getting recent activity logs: {e}")
        return []

def get_all_activity_logs():
    """Get all activity logs (for initial page load)"""
    try:
        with activity_logs_lock:
            # Return all logs for initial load
            return activity_logs[:]
    except Exception as e:
        print(f"Error getting all activity logs: {e}")
        return []

def clear_activity_logs():
    """Clear all activity logs"""
    try:
        with activity_logs_lock:
            activity_logs.clear()
        with client_sessions_lock:
            client_last_log_index.clear()
        return True
    except Exception as e:
        print(f"Error clearing activity logs: {e}")
        return False

def log_activity_detection_simple(detections_dict):
    """Simple function to log activity detections for testing"""
    try:
        if not detections_dict:
            return
        
        timestamp = datetime.datetime.now()
        
        with activity_logs_lock:
            for employee_id, actions in detections_dict.items():
                if actions:  # Only log if there are actions
                    log_entry = {
                        "timestamp": timestamp.isoformat(),
                        "employee_id": employee_id,
                        "actions": actions,
                        "confidence_scores": [0.8] * len(actions)  # Dummy confidence scores
                    }
                    
                    activity_logs.append(log_entry)
                    print(f"Added test activity log: {employee_id} - {actions}")
            
            # No longer truncate activity_logs, allow it to grow unbounded
            # if len(activity_logs) > 100:
            #     activity_logs[:] = activity_logs[-100:]
                
    except Exception as e:
        print(f"Error logging simple activity detection: {e}")

# ----------------------
# Main Application
# ----------------------
# Create global instances for use in Flask application
config = Config()
model_manager = ModelManager(config)
video_manager = VideoManager(config, model_manager)

def start_background_video_thread():
    """Start the video processing in a background thread"""
    return video_manager.start_background_video_thread()

def get_latest_frame():
    """Get the latest processed frame"""
    return video_manager.get_latest_frame()

def toggle_activity_recognition():
    """Toggle the activity recognition feature on/off"""
    return model_manager.toggle_activity_recognition()

def get_activity_recognition_state():
    """Get current state of activity recognition"""
    return model_manager.get_activity_recognition_state()

def start_recording(recording_type="original"):
    """Start recording of the specified type"""
    video_manager.setup_recording(recording_type)
    return video_manager.original_recording_active if recording_type == "original" else video_manager.segmented_recording_active

def stop_recording(recording_type="original"):
    """Stop recording of the specified type"""
    video_manager.stop_recording(recording_type)
    return not (video_manager.original_recording_active if recording_type == "original" else video_manager.segmented_recording_active)

def get_recording_status():
    """Get the status of both recording types"""
    return {
        "original": video_manager.original_recording_active,
        "segmented": video_manager.segmented_recording_active
    }

def start_dual_recording():
    """Start both original and segmented recordings simultaneously"""
    original_success = start_recording('original')
    segmented_success = start_recording('segmented')
    return original_success and segmented_success

def stop_dual_recording():
    """Stop both original and segmented recordings simultaneously"""
    stop_recording('original')
    stop_recording('segmented')
    return True

# ----------------------
# Employee Activity Monitoring Functions
# ----------------------

def capture_employee_activity():
    """Capture current frame and detected employees for activity monitoring"""
    try:
        # Get current frame and detections from video manager
        if not hasattr(video_manager, 'current_frame') or video_manager.current_frame is None:
            return {"success": False, "error": "No current frame available"}
            
        frame = video_manager.current_frame.copy()
        
        # Get current detections and activities
        detections = getattr(video_manager, 'current_detections', [])
        activities = getattr(video_manager, 'current_activities', {})
        
        if not detections:
            return {"success": False, "error": "No employees detected in current frame"}
        
        captures = []
        timestamp = datetime.datetime.now()
        
        for i, detection in enumerate(detections):
            # detection format: [x1, y1, x2, y2, confidence]
            x1, y1, x2, y2, conf = detection
            
            # Check if person is in ROI
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            if is_on_active_side((cx, cy), config.roi_line_p1, config.roi_line_p2):
                # Crop employee image
                employee_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if employee_crop.size > 0:
                    # Generate employee ID
                    employee_id = f"EMP_{i+1:03d}"
                    
                    # Create timestamped folder
                    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
                    capture_dir = os.path.join(employee_act_dir, timestamp_str)
                    os.makedirs(capture_dir, exist_ok=True)
                      # Save image
                    filename = f"employee_{employee_id}_{timestamp_str}.jpg"
                    filepath = os.path.join(capture_dir, filename)
                    cv2.imwrite(filepath, employee_crop)
                    
                    # Get activities for this employee
                    employee_activities = activities.get(i, [])
                    
                    # Store capture info
                    capture_info = {
                        "employee_id": employee_id,
                        "timestamp": timestamp.isoformat(),
                        "image_path": os.path.join(timestamp_str, filename),
                        "activities": employee_activities,
                        "roi_info": f"({cx}, {cy})",
                        "confidence": conf
                    }
                    
                    # Save metadata as JSON file alongside the image
                    metadata_filename = f"employee_{employee_id}_{timestamp_str}.json"
                    metadata_filepath = os.path.join(capture_dir, metadata_filename)
                    import json
                    with open(metadata_filepath, 'w') as f:
                        json.dump(capture_info, f, indent=2)
                    
                    captures.append(capture_info)
                    employee_captures.append(capture_info)
        
        # No longer truncate employee_captures, allow it to grow unbounded
        # if len(employee_captures) > 50:
        #     employee_captures[:] = employee_captures[-50:]
        
        result = {
            "success": True,
            "captures_count": len(captures),
            "captures": captures,
            "timestamp": timestamp.isoformat()
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_employee_captures():
    """Get list of employee captures with statistics"""
    try:
        # Calculate statistics
        unique_employees = set()
        for capture in employee_captures:
            unique_employees.add(capture["employee_id"])
        
        statistics = {
            "total_employees": len(unique_employees),
            "total_captures": len(employee_captures),
            "last_capture": employee_captures[-1]["timestamp"] if employee_captures else None
        }
        
        return {
            "success": True,
            "captures": employee_captures,
            "statistics": statistics
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "captures": [], "statistics": {}}

def clear_employee_captures():
    """Clear all employee captures and images"""
    try:
        # Clear in-memory captures
        employee_captures.clear()
        
        # Optionally clear image files (commented out for safety)
        # import shutil
        # if os.path.exists(employee_act_dir):
        #     for item in os.listdir(employee_act_dir):
        #         item_path = os.path.join(employee_act_dir, item)
        #         if os.path.isdir(item_path):
        #             shutil.rmtree(item_path)
        
        return {"success": True, "message": "Employee captures cleared"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_active_employees_count():
    """Get count of currently active employees"""
    try:
        # Get current detections
        detections = getattr(video_manager, 'current_detections', [])
        active_count = 0
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            # Count only employees in ROI
            if is_on_active_side((cx, cy), config.roi_line_p1, config.roi_line_p2):
                active_count += 1
        return {"active_employees": active_count}
    except Exception as e:
        return {"active_employees": 0}

def get_all_employee_captures_with_metadata():
    """Get all employee captures with metadata from stored JSON files"""
    try:
        import json
        import glob
        all_captures = []
        
        # Walk through all subdirectories in employee_act_dir
        for root, dirs, files in os.walk(employee_act_dir):
            # Look for JSON metadata files
            json_files = [f for f in files if f.endswith('.json')]
            
            for json_file in json_files:
                json_path = os.path.join(root, json_file)
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify corresponding image exists
                    image_filename = json_file.replace('.json', '.jpg')
                    image_path = os.path.join(root, image_filename)
                    
                    if os.path.exists(image_path):
                        all_captures.append(metadata)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error reading metadata from {json_path}: {e}")
                    continue
        
        # Sort by timestamp (newest first)
        all_captures.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Calculate statistics
        unique_employees = set()
        for capture in all_captures:
            unique_employees.add(capture.get("employee_id", "Unknown"))
        
        statistics = {
            "total_employees": len(unique_employees),
            "total_captures": len(all_captures),
            "last_capture": all_captures[0]["timestamp"] if all_captures else None
        }
        
        return {
            "success": True,
            "captures": all_captures,
            "statistics": statistics
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "captures": [], "statistics": {}}

# --- User Session Tracking for Connected User Count ---
import threading
from flask import session, jsonify, request

active_sessions = set()
active_sessions_lock = threading.Lock()

def add_active_session(session_id):
    with active_sessions_lock:
        active_sessions.add(session_id)

def remove_active_session(session_id):
    with active_sessions_lock:
        active_sessions.discard(session_id)

def get_active_sessions_count():
    with active_sessions_lock:
        return len(active_sessions)

def is_session_active(session_id):
    with active_sessions_lock:
        return session_id in active_sessions

# Flask API endpoint to get the number of connected users
# (This should be registered in app.py, but logic is here for import)
def connected_users_api():
    return jsonify({"connected_users": get_active_sessions_count()})
