"""
Hand Gesture Presentation Controller V3
Real-time gesture recognition for controlling presentations using Canny+HOG features.
"""

import cv2
import numpy as np
import joblib
import time
import pyautogui
from skimage.feature import hog
from pathlib import Path
import statistics
from collections import deque

###############################################################################
# CONFIGURATION SECTION
###############################################################################

class GestureConfig:
    """Configuration for Hand Gesture Presentation Controller"""
    
    # ========== MODEL SETTINGS ==========
    MODEL_PATH = "artifacts/gesture_svm_v3.pkl"  
    BACKUP_MODEL_PATH = "artifacts/gesture_svm.pkl"  
    
    # ========== CANNY EDGE PARAMETERS ==========
    CANNY_PARAMS = {
        "threshold1": 50,  
        "threshold2": 150,  
        "apertureSize": 3,
        "L2gradient": True,
    }

    # ========== HOG PARAMETERS ==========
    HOG_PARAMS = {
        "orientations": 9,
        "pixels_per_cell": (16, 16),  
        "cells_per_block": (2, 2),
        "transform_sqrt": True,
        "block_norm": "L2-Hys",
        "feature_vector": True,
    }
    
    TARGET_IMAGE_SIZE = (128, 128)
    
    # ========== GESTURE MAPPING ==========
    # Map model predictions to consistent action labels
    # IMPORTANT: This should match the labels your model was trained with
    # Model might output "next", "prev", "back", etc.
    LABEL_MAP = {
        "next": "next",           # "next" gesture stays as "next"
        "prev": "previous",       # "prev" gesture maps to "previous"
        "back": "previous",       # "back" gesture maps to "previous"
        "previous": "previous"    # "previous" gesture stays as "previous"
    }
    
    # ========== ACTION MAPPING ==========
    # Map action labels to keyboard keys
    ACTION_MAP = {
        "next": "right",        # Right arrow key for next slide
        "previous": "left"      # Left arrow key for previous slide
    }
    
    # ========== INFERENCE SETTINGS ==========
    CONFIDENCE_THRESHOLDS = {
        "next": 0.80,      # 80% confidence for next
        "previous": 0.95,  # 95% confidence for previous
        "default": 0.70    # Default for any other gestures
    }
    
    COOLDOWN_SECONDS = 2.0
    
    # ========== HAND DETECTION SETTINGS ==========
    USE_MEDIAPIPE = True
    AUTO_ROI_PADDING = 30
    MIN_HAND_SIZE = 80
    
    # ========== NEW HAND DETECTION SETTINGS ==========
    HAND_STABILITY_THRESHOLD = 5  # Frames a hand must be stable before switching
    NEW_HAND_DISTANCE_THRESHOLD = 100  # Pixels - minimum distance to consider as new hand
    HAND_TRACKING_ENABLED = True  # Enable/disable hand tracking
    
    # ========== SMOOTHING SETTINGS ==========
    SMOOTHING_FRAMES = 3
    CONSENSUS_REQUIRED = "majority"
    
    # ========== DISPLAY SETTINGS ==========
    DISPLAY_FPS = True
    SHOW_HISTORY = True
    SHOW_CONFIDENCE = True
    SHOW_INSTRUCTIONS = True
    SHOW_MODEL_INFO = True
    SHOW_KEY_INFO = True
    SHOW_ALL_PREDICTIONS = True  # Show all prediction probabilities
    SHOW_HAND_TRACKING_INFO = True  # Show hand tracking information
    
    # ========== VISUAL APPEARANCE ==========
    ROI_COLOR_HIGH_CONF = (0, 255, 0)
    ROI_COLOR_LOW_CONF = (0, 165, 255)
    ROI_COLOR_NEW_HAND = (255, 0, 255)  # Magenta for new hand
    ROI_COLOR_OLD_HAND = (255, 255, 0)  # Yellow for old hand
    TEXT_COLOR = (255, 255, 255)
    WINDOW_NAME = "Hand Gesture Controller"
    
    # ========== CAMERA SETTINGS ==========
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FLIP_HORIZONTAL = True
    
    # ========== CONTROL KEYS ==========
    KEY_QUIT = 'q'
    KEY_PAUSE = 'p'
    KEY_DEBUG = 'd'  # Toggle debug mode
    KEY_TRACKING = 't'  # Toggle hand tracking
    
    # ========== DEBUG SETTINGS ==========
    VERBOSE = True
    PRINT_PREDICTIONS = True  # Changed to True for debugging
    PRINT_MODEL_INFO = True
    PRINT_ACTIONS = True
    DEBUG_MODE = False  # Show detailed prediction info

###############################################################################
# HAND TRACKING CLASS
###############################################################################

class HandTracker:
    """Tracks multiple hands and identifies the newest/most active hand"""
    
    def __init__(self, config):
        self.cfg = config
        self.tracked_hands = {}  # hand_id: {roi, last_seen, stability_count, first_seen}
        self.current_hand_id = None
        self.next_hand_id = 0
        self.hand_stability_counter = 0
        self.last_hand_center = None
        
    def calculate_center(self, roi):
        """Calculate center point of ROI"""
        x0, y0, x1, y1 = roi
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        return (center_x, center_y)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if point1 is None or point2 is None:
            return float('inf')
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def find_closest_hand(self, current_center, exclude_id=None):
        """Find the closest existing hand to the current center"""
        closest_id = None
        min_distance = float('inf')
        
        for hand_id, hand_data in self.tracked_hands.items():
            if exclude_id is not None and hand_id == exclude_id:
                continue
                
            if 'center' in hand_data:
                distance = self.calculate_distance(current_center, hand_data['center'])
                if distance < min_distance:
                    min_distance = distance
                    closest_id = hand_id
        
        return closest_id, min_distance
    
    def update_tracked_hands(self, current_roi):
        """Update tracked hands and identify the newest hand"""
        if current_roi is None:
            # No hand detected, clear stability counter
            self.hand_stability_counter = 0
            return None
        
        current_center = self.calculate_center(current_roi)
        current_time = time.time()
        
        # Find closest existing hand
        closest_id, min_distance = self.find_closest_hand(current_center)
        
        # Check if this is a new hand
        if closest_id is None or min_distance > self.cfg.NEW_HAND_DISTANCE_THRESHOLD:
            # This might be a new hand, wait for stability
            if self.last_hand_center is not None:
                distance_from_last = self.calculate_distance(current_center, self.last_hand_center)
                
                # If hand is stable (not moving much)
                if distance_from_last < 20:  # Small movement threshold
                    self.hand_stability_counter += 1
                else:
                    self.hand_stability_counter = 0
                    self.last_hand_center = current_center
            else:
                self.last_hand_center = current_center
                self.hand_stability_counter = 1
            
            # If hand has been stable enough, register as new hand
            if self.hand_stability_counter >= self.cfg.HAND_STABILITY_THRESHOLD:
                # Create new hand entry
                new_id = self.next_hand_id
                self.tracked_hands[new_id] = {
                    'roi': current_roi,
                    'center': current_center,
                    'last_seen': current_time,
                    'stability_count': self.hand_stability_counter,
                    'first_seen': current_time,
                    'age': 0  # in seconds
                }
                self.next_hand_id += 1
                self.current_hand_id = new_id
                self.hand_stability_counter = 0
                return new_id
            else:
                # Still stabilizing, return None or closest existing
                return self.current_hand_id if self.current_hand_id is not None else closest_id
        else:
            # Update existing hand
            self.tracked_hands[closest_id].update({
                'roi': current_roi,
                'center': current_center,
                'last_seen': current_time,
                'age': current_time - self.tracked_hands[closest_id]['first_seen']
            })
            
            # Check if we should switch to this hand (if it's newer)
            if self.current_hand_id is None:
                self.current_hand_id = closest_id
            else:
                # Switch to newer hand if it was detected more recently
                current_hand_data = self.tracked_hands.get(self.current_hand_id)
                new_hand_data = self.tracked_hands[closest_id]
                
                if (new_hand_data['first_seen'] > current_hand_data['first_seen'] and 
                    current_time - new_hand_data['first_seen'] < 2.0):  # New within last 2 seconds
                    self.current_hand_id = closest_id
            
            return self.current_hand_id
    
    def get_current_roi(self):
        """Get ROI of current hand"""
        if self.current_hand_id is not None and self.current_hand_id in self.tracked_hands:
            return self.tracked_hands[self.current_hand_id]['roi']
        return None
    
    def get_hand_info(self, hand_id):
        """Get information about a specific hand"""
        if hand_id in self.tracked_hands:
            return self.tracked_hands[hand_id]
        return None
    
    def get_all_hands(self):
        """Get all tracked hands"""
        return self.tracked_hands
    
    def cleanup_old_hands(self, max_age_seconds=10):
        """Remove hands that haven't been seen for a while"""
        current_time = time.time()
        hands_to_remove = []
        
        for hand_id, hand_data in self.tracked_hands.items():
            if current_time - hand_data['last_seen'] > max_age_seconds:
                hands_to_remove.append(hand_id)
        
        for hand_id in hands_to_remove:
            if hand_id == self.current_hand_id:
                self.current_hand_id = None
            del self.tracked_hands[hand_id]
    
    def reset(self):
        """Reset all tracking"""
        self.tracked_hands.clear()
        self.current_hand_id = None
        self.hand_stability_counter = 0
        self.last_hand_center = None

###############################################################################
# MAIN GESTURE CONTROLLER CLASS
###############################################################################

class GestureController:
    def __init__(self, config_obj=None):
        """
        Initialize the Gesture Controller for real-time hand gesture recognition.
        """
        self.cfg = config_obj if config_obj else GestureConfig
        
        # State variables
        self.hand_detector = None
        self.current_roi = None
        self.last_good_roi = None
        self.hand_detected = False
        self.last_trigger = 0.0
        self.cap = None
        self.fps = 0
        self.paused = False
        self.debug_mode = self.cfg.DEBUG_MODE
        
        # Initialize hand tracker
        self.hand_tracker = HandTracker(self.cfg)
        self.current_hand_id = None
        
        # Model information
        self.model_params = None
        self.model_kernel = None
        self.model_classes = []  # Store model's class names
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=self.cfg.SMOOTHING_FRAMES)
        
        # Print configuration
        if self.cfg.VERBOSE:
            self.print_basic_info()
        
        # Load the trained model
        self.load_model()
    
    def print_basic_info(self):
        """Print basic startup information."""
        print("="*60)
        print("HAND GESTURE CONTROLLER")
        print("="*60)
        print(f"Model: {self.cfg.MODEL_PATH}")
        print(f"Camera: {self.cfg.CAMERA_ID}")
        print(f"Hand Detection: {'MediaPipe' if self.cfg.USE_MEDIAPIPE else 'Skin Color'}")
        print(f"Hand Tracking: {'Enabled' if self.cfg.HAND_TRACKING_ENABLED else 'Disabled'}")
        print(f"Controls:")
        print(f"  • Next Slide: Right Arrow Key (→)")
        print(f"  • Previous Slide: Left Arrow Key (←)")
        print(f"  • Pause/Resume: Press '{self.cfg.KEY_PAUSE}'")
        print(f"  • Debug Mode: Press '{self.cfg.KEY_DEBUG}'")
        print(f"  • Tracking Toggle: Press '{self.cfg.KEY_TRACKING}'")
        print(f"  • Quit: Press '{self.cfg.KEY_QUIT}'")
        print(f"Label Mapping: {self.cfg.LABEL_MAP}")
        print("="*60)
    
    def get_confidence_threshold(self, label):
        """Get confidence threshold for specific gesture label."""
        # First try exact match
        if label in self.cfg.CONFIDENCE_THRESHOLDS:
            return self.cfg.CONFIDENCE_THRESHOLDS[label]
        
        # Try mapped label
        mapped_label = self.cfg.LABEL_MAP.get(label, label)
        if mapped_label in self.cfg.CONFIDENCE_THRESHOLDS:
            return self.cfg.CONFIDENCE_THRESHOLDS[mapped_label]
        
        # Default threshold
        return self.cfg.CONFIDENCE_THRESHOLDS["default"]
    
    def load_model(self):
        """Load the trained SVM model and preprocessing parameters."""
        model_path = Path(self.cfg.MODEL_PATH) 
        
        try:
            bundle = joblib.load(model_path)
            print(f"✓ Model loaded successfully")
            
            # Extract model components
            self.clf = bundle["model"]
            self.target_size = tuple(bundle.get("target_size", self.cfg.TARGET_IMAGE_SIZE))
            
            # Store model classes
            self.model_classes = list(self.clf.classes_)
            
            # Get kernel type from the model
            self.model_kernel = getattr(self.clf, 'kernel', 'unknown')
            
            if self.cfg.PRINT_MODEL_INFO:
                print(f"✓ Model Details:")
                print(f"  • Type: {self.clf.__class__.__name__}")
                print(f"  • Classes: {self.model_classes}")
                print(f"  • # Features: {self.clf.n_features_in_}")
                
        except FileNotFoundError:
            backup_path = Path(self.cfg.BACKUP_MODEL_PATH)
            try:
                bundle = joblib.load(backup_path)
                print(f"✓ Backup model loaded")
                self.clf = bundle["model"]
                self.target_size = tuple(bundle.get("target_size", self.cfg.TARGET_IMAGE_SIZE))
                self.model_kernel = getattr(self.clf, 'kernel', 'unknown')
                self.model_classes = list(self.clf.classes_)
            except FileNotFoundError:
                print(f"❌ Error: No model found")
                print("Please run the training notebook first.")
                raise
        
        # Use parameters from config
        self.hog_params = self.cfg.HOG_PARAMS.copy()
        self.canny_params = self.cfg.CANNY_PARAMS.copy()
    
    def init_hand_detector(self):
        """Initialize hand detector based on configuration."""
        if self.cfg.USE_MEDIAPIPE:
            try:
                import mediapipe as mp 
                self.mp_hands = mp.solutions.hands
                self.hand_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,  # Allow detecting multiple hands
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                if self.cfg.VERBOSE:
                    print("✓ MediaPipe hand detector initialized")
                return True
            except ImportError:
                print("⚠️ MediaPipe not installed. Using skin detection.")
                print("   Install with: pip install mediapipe")
                self.cfg.USE_MEDIAPIPE = False 
                return self.init_hand_detector()
        
        if self.cfg.VERBOSE:
            print("✓ Using OpenCV skin detection")
        return True
    
    def detect_hand_with_mediapipe(self, frame):
        """Detect hand using MediaPipe and return the newest hand."""
        if not self.hand_detector:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get all detected hands
            all_rois = []
            h, w = frame.shape[:2]
            
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # Add padding
                roi = self._adjust_roi_with_constraints(x_min, y_min, x_max, y_max, frame.shape)
                if roi:
                    all_rois.append(roi)
            
            if all_rois:
                # If hand tracking is enabled, use tracker to find newest hand
                if self.cfg.HAND_TRACKING_ENABLED:
                    # For each ROI, update tracker
                    newest_roi = None
                    for roi in all_rois:
                        hand_id = self.hand_tracker.update_tracked_hands(roi)
                        if hand_id is not None:
                            hand_info = self.hand_tracker.get_hand_info(hand_id)
                            if hand_info and hand_info.get('first_seen', 0) > 0:
                                if newest_roi is None or hand_info['first_seen'] > newest_roi[1]:
                                    newest_roi = (roi, hand_info['first_seen'], hand_id)
                    
                    if newest_roi:
                        self.current_hand_id = newest_roi[2]
                        self.hand_detected = True
                        return newest_roi[0]
                else:
                    # Simple mode: just use the first (largest) hand
                    roi = all_rois[0]
                    self.last_good_roi = roi
                    self.hand_detected = True
                    return roi
        
        self.hand_detected = False
        return None
    
    def detect_hand_with_skincolor(self, frame):
        """Detect hand using skin color detection."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Use the largest contour
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Skip if too small
            if w < 50 or h < 50:
                self.hand_detected = False
                return None
            
            # Add padding
            x_min = max(0, x - self.cfg.AUTO_ROI_PADDING)
            y_min = max(0, y - self.cfg.AUTO_ROI_PADDING)
            x_max = min(frame.shape[1], x + w + self.cfg.AUTO_ROI_PADDING)
            y_max = min(frame.shape[0], y + h + self.cfg.AUTO_ROI_PADDING)
            
            roi = (x_min, y_min, x_max, y_max)
            
            # Update hand tracker if enabled
            if self.cfg.HAND_TRACKING_ENABLED:
                hand_id = self.hand_tracker.update_tracked_hands(roi)
                if hand_id is not None:
                    self.current_hand_id = hand_id
            
            self.last_good_roi = roi
            self.hand_detected = True
            return roi
        
        self.hand_detected = False
        return None
    
    def _adjust_roi_with_constraints(self, x_min, y_min, x_max, y_max, frame_shape):
        """Adjust ROI with padding and minimum size constraints."""
        h, w = frame_shape[:2]
        
        # Add padding
        x_min = max(0, x_min - self.cfg.AUTO_ROI_PADDING)
        y_min = max(0, y_min - self.cfg.AUTO_ROI_PADDING)
        x_max = min(w, x_max + self.cfg.AUTO_ROI_PADDING)
        y_max = min(h, y_max + self.cfg.AUTO_ROI_PADDING)
        
        # Ensure minimum size
        if (x_max - x_min) < self.cfg.MIN_HAND_SIZE:
            center_x = (x_min + x_max) // 2
            half_size = self.cfg.MIN_HAND_SIZE // 2
            x_min = max(0, center_x - half_size)
            x_max = min(w, center_x + half_size)
        
        if (y_max - y_min) < self.cfg.MIN_HAND_SIZE:
            center_y = (y_min + y_max) // 2
            half_size = self.cfg.MIN_HAND_SIZE // 2
            y_min = max(0, center_y - half_size)
            y_max = min(h, center_y + half_size)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_current_roi(self, frame):
        """Get ROI of the newest detected hand."""
        if self.hand_detector and self.cfg.USE_MEDIAPIPE:
            roi = self.detect_hand_with_mediapipe(frame)
        else:
            roi = self.detect_hand_with_skincolor(frame)
        
        self.current_roi = roi
        
        # Clean up old hands periodically
        if self.cfg.HAND_TRACKING_ENABLED and self.cfg.VERBOSE:
            self.hand_tracker.cleanup_old_hands()
        
        return roi
    
    def preprocess_frame(self, frame, roi_bounds):
        """Preprocess the ROI frame for inference using Canny+HOG features."""
        if roi_bounds is None:
            return None
        
        x0, y0, x1, y1 = roi_bounds
        
        # Validate ROI
        if x1 <= x0 or y1 <= y0:
            return None
        
        # Extract ROI
        roi_frame = frame[y0:y1, x0:x1]
        if roi_frame.size == 0:
            return None
        
        # 1. Convert to grayscale and resize
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, self.target_size)
        
        # 2. Apply Canny Edge detection
        edges = cv2.Canny(
            resized_gray,
            threshold1=self.canny_params["threshold1"],
            threshold2=self.canny_params["threshold2"],
            apertureSize=self.canny_params["apertureSize"],
            L2gradient=self.canny_params["L2gradient"]
        )
        
        # 3. Normalize edge map
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # 4. Compute HOG on the edge-detected image
        hog_features = hog(
            edges_normalized,
            orientations=self.hog_params["orientations"],
            pixels_per_cell=self.hog_params["pixels_per_cell"],
            cells_per_block=self.hog_params["cells_per_block"],
            transform_sqrt=self.hog_params["transform_sqrt"],
            block_norm=self.hog_params["block_norm"],
            feature_vector=self.hog_params["feature_vector"]
        )
        
        # 5. Flatten the Canny edge map
        canny_flat = edges.flatten().astype(np.float32) / 255.0
        
        # 6. Concatenate both feature sets
        combined_features = np.concatenate([hog_features, canny_flat])
        
        # Ensure dimension matches
        if combined_features.shape[0] != self.clf.n_features_in_:
            # Pad or truncate to match
            if combined_features.shape[0] > self.clf.n_features_in_:
                combined_features = combined_features[:self.clf.n_features_in_]
            else:
                padding = np.zeros(self.clf.n_features_in_ - combined_features.shape[0])
                combined_features = np.concatenate([combined_features, padding])
        
        return combined_features.reshape(1, -1)
    
    def get_consensus_prediction(self):
        """Get consensus prediction from history."""
        if len(self.prediction_history) < self.cfg.SMOOTHING_FRAMES:
            return None
        
        if self.cfg.CONSENSUS_REQUIRED == "all":
            # All frames must agree
            first_pred = self.prediction_history[0]
            if all(pred == first_pred for pred in self.prediction_history):
                return first_pred if first_pred != "neutral" else None
        elif self.cfg.CONSENSUS_REQUIRED == "majority":
            # Majority vote
            try:
                most_common = statistics.mode(self.prediction_history)
                count = self.prediction_history.count(most_common)
                if count > len(self.prediction_history) // 2 and most_common != "neutral":
                    return most_common
            except statistics.StatisticsError:
                pass
        
        return None
    
    def _draw_hand_tracking_info(self, frame):
        """Draw hand tracking information on frame."""
        if not self.cfg.SHOW_HAND_TRACKING_INFO or not self.cfg.HAND_TRACKING_ENABLED:
            return
        
        y_offset = frame.shape[0] - 200
        line_height = 18
        
        # Draw header
        cv2.putText(
            frame,
            "Hand Tracking:",
            (frame.shape[1] - 200, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_offset += line_height
        
        # Draw tracked hands info
        all_hands = self.hand_tracker.get_all_hands()
        
        if not all_hands:
            cv2.putText(
                frame,
                "No hands tracked",
                (frame.shape[1] - 200, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            y_offset += line_height
        else:
            current_time = time.time()
            for hand_id, hand_data in all_hands.items():
                age = current_time - hand_data['first_seen']
                last_seen = current_time - hand_data['last_seen']
                
                # Color code: current hand in green, others in blue
                if hand_id == self.current_hand_id:
                    color = (0, 255, 0)
                    prefix = "→ "
                else:
                    color = (100, 100, 255)
                    prefix = "  "
                
                hand_info = f"{prefix}Hand {hand_id}: {age:.1f}s old"
                cv2.putText(
                    frame,
                    hand_info,
                    (frame.shape[1] - 200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                y_offset += line_height
        
        # Draw current hand info
        if self.current_hand_id is not None:
            current_hand_info = self.hand_tracker.get_hand_info(self.current_hand_id)
            if current_hand_info:
                age = current_time - current_hand_info['first_seen']
                cv2.putText(
                    frame,
                    f"Active: Hand {self.current_hand_id} ({age:.1f}s)",
                    (frame.shape[1] - 200, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                )
    
    def predict_gesture(self, frame):
        """Predict gesture from frame with smoothing."""
        display_frame = frame.copy()
        
        # Get ROI
        current_roi = self.get_current_roi(frame)
        
        # Initialize values
        current_label = None
        confidence = 0.0
        confidence_threshold = 0.0
        all_predictions = {}
        
        if current_roi and self.hand_detected:
            # Extract features and predict
            descriptor = self.preprocess_frame(frame, current_roi)
            
            if descriptor is not None:
                # Get probabilities for all classes
                proba = self.clf.predict_proba(descriptor)[0]
                top_idx = int(np.argmax(proba))
                current_label = self.clf.classes_[top_idx]
                confidence = float(proba[top_idx])
                
                # Store all predictions for debugging
                for i, label in enumerate(self.clf.classes_):
                    all_predictions[label] = float(proba[i])
                
                # Get the appropriate confidence threshold for this gesture
                # Use mapped label for threshold lookup
                mapped_label = self.cfg.LABEL_MAP.get(current_label, current_label)
                confidence_threshold = self.get_confidence_threshold(mapped_label)
                
                # Debug output
                if self.cfg.PRINT_PREDICTIONS or self.debug_mode:
                    print(f"\n=== Prediction ===")
                    print(f"Raw model output: {current_label} ({confidence:.3f})")
                    print(f"Mapped to: {mapped_label}")
                    print(f"Required confidence: {confidence_threshold:.3f}")
                    print(f"All predictions:")
                    for label, prob in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True):
                        mapped = self.cfg.LABEL_MAP.get(label, label)
                        action = self.cfg.ACTION_MAP.get(mapped, "None")
                        print(f"  {label:10s} -> {mapped:10s} -> {action:5s}: {prob:.3f}")
            
            # Update history - use mapped label
            if confidence > confidence_threshold:
                mapped_label = self.cfg.LABEL_MAP.get(current_label, current_label)
                self.prediction_history.append(mapped_label)
            else:
                self.prediction_history.append("neutral")
            
            # Draw ROI and info
            self._draw_roi_and_info(display_frame, current_roi, current_label, confidence, confidence_threshold)
            
            # Draw debug info if enabled
            if self.debug_mode:
                self._draw_debug_info(display_frame, all_predictions)
        else:
            # No hand detected
            self.prediction_history.clear()
            if not self.paused:
                self._draw_no_hand_message(display_frame)
        
        # Get consensus prediction
        consensus_label = self.get_consensus_prediction()
        
        # Draw UI elements
        self._draw_ui_elements(display_frame, consensus_label)
        
        # Draw hand tracking info
        self._draw_hand_tracking_info(display_frame)
        
        return consensus_label, confidence, display_frame
    
    def _draw_roi_and_info(self, frame, roi, label, confidence, threshold):
        """Draw ROI and prediction info on frame."""
        x0, y0, x1, y1 = roi
        
        # Map label for display
        mapped_label = self.cfg.LABEL_MAP.get(label, label)
        
        # Choose color based on hand age and confidence
        if self.cfg.HAND_TRACKING_ENABLED and self.current_hand_id is not None:
            hand_info = self.hand_tracker.get_hand_info(self.current_hand_id)
            if hand_info:
                age = time.time() - hand_info['first_seen']
                if age < 2.0:  # Very new hand (less than 2 seconds)
                    color = self.cfg.ROI_COLOR_NEW_HAND
                elif age < 5.0:  # Relatively new hand
                    color = (255, 150, 0)  # Orange
                else:
                    if confidence >= threshold:
                        color = self.cfg.ROI_COLOR_HIGH_CONF
                    else:
                        color = self.cfg.ROI_COLOR_LOW_CONF
            else:
                if confidence >= threshold:
                    color = self.cfg.ROI_COLOR_HIGH_CONF
                else:
                    color = self.cfg.ROI_COLOR_LOW_CONF
        else:
            # Default color scheme without tracking
            if confidence >= threshold:
                color = self.cfg.ROI_COLOR_HIGH_CONF
            else:
                color = self.cfg.ROI_COLOR_LOW_CONF
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Draw ROI border
        border_thickness = 3 if confidence >= threshold else 2
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, border_thickness)
        
        # Draw label and confidence with threshold info
        if self.cfg.SHOW_CONFIDENCE:
            label_text = f"{mapped_label}: {confidence:.2f} (req: {threshold:.2f})"
        else:
            label_text = f"{mapped_label}"
        
        cv2.putText(
            frame,
            label_text,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        
        # Draw hand age if tracking is enabled
        if self.cfg.HAND_TRACKING_ENABLED and self.current_hand_id is not None:
            hand_info = self.hand_tracker.get_hand_info(self.current_hand_id)
            if hand_info:
                age = time.time() - hand_info['first_seen']
                age_text = f"Age: {age:.1f}s"
                cv2.putText(
                    frame,
                    age_text,
                    (x0, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
    
    def _draw_debug_info(self, frame, predictions):
        """Draw debug prediction information."""
        y_offset = frame.shape[0] - 150
        line_height = 20
        
        cv2.putText(
            frame,
            "DEBUG MODE - All Predictions:",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        y_offset += line_height
        
        # Sort predictions by probability
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_preds:
            mapped_label = self.cfg.LABEL_MAP.get(label, label)
            action = self.cfg.ACTION_MAP.get(mapped_label, "None")
            color = (0, 255, 0) if prob > 0.5 else (200, 200, 200)
            
            text = f"{label} -> {mapped_label} -> {action}: {prob:.3f}"
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            y_offset += line_height
    
    def _draw_no_hand_message(self, frame):
        """Draw message when no hand is detected."""
        h, w = frame.shape[:2]
        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (w//2 - 100, h//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        
        cv2.putText(
            frame,
            "Show your hand to begin",
            (w//2 - 120, h//2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.cfg.TEXT_COLOR,
            1,
        )
    
    def _draw_ui_elements(self, frame, consensus_label):
        """Draw UI elements on frame."""
        y_offset = 30
        line_height = 25
        
        # Paused state
        if self.paused:
            cv2.putText(
                frame,
                "PAUSED",
                (frame.shape[1]//2 - 50, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            y_offset += line_height * 2
        
        # Detection method
        method = "MediaPipe" if (self.hand_detector and self.cfg.USE_MEDIAPIPE) else "Skin"
        cv2.putText(
            frame,
            f"Detection: {method}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.cfg.TEXT_COLOR,
            1,
        )
        y_offset += line_height
        
        # Hand status
        status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if self.hand_detected else "NO HAND"
        cv2.putText(
            frame,
            f"Status: {status_text}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            1,
        )
        y_offset += line_height
        
        # Model classes
        if self.cfg.SHOW_MODEL_INFO:
            classes_str = ", ".join(self.model_classes)
            cv2.putText(
                frame,
                f"Model classes: {classes_str}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )
            y_offset += line_height
        
        # Key mapping info
        if self.cfg.SHOW_KEY_INFO:
            cv2.putText(
                frame,
                "Controls: Next=→, Prev=←",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.cfg.TEXT_COLOR,
                1,
            )
            y_offset += line_height
        
        # Prediction history
        if self.cfg.SHOW_HISTORY and len(self.prediction_history) > 0:
            hist_str = " ".join([p[0].upper() for p in self.prediction_history])
            cv2.putText(
                frame,
                f"History: [{hist_str}]",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.cfg.TEXT_COLOR,
                1,
            )
            y_offset += line_height
        
        # Consensus
        if consensus_label and not self.paused:
            # Get the key that will be pressed
            key_to_press = self.cfg.ACTION_MAP.get(consensus_label, "?")
            arrow_symbol = "→" if key_to_press == "right" else "←"
            
            cv2.putText(
                frame,
                f"Action: {consensus_label} {arrow_symbol}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            y_offset += line_height
        
        # Debug mode indicator
        if self.debug_mode:
            cv2.putText(
                frame,
                "DEBUG MODE",
                (frame.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
        
        # Hand tracking status
        if self.cfg.HAND_TRACKING_ENABLED:
            cv2.putText(
                frame,
                "TRACKING: ON",
                (frame.shape[1] - 120, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        
        # Instructions
        if self.cfg.SHOW_INSTRUCTIONS:
            cv2.putText(
                frame,
                f"Press '{self.cfg.KEY_PAUSE}'=pause | '{self.cfg.KEY_DEBUG}'=debug | '{self.cfg.KEY_TRACKING}'=track | '{self.cfg.KEY_QUIT}'=quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.cfg.TEXT_COLOR,
                1,
            )
        
        # FPS
        if self.cfg.DISPLAY_FPS and hasattr(self, 'fps'):
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (frame.shape[1] - 100, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                self.cfg.TEXT_COLOR,
                2,
            )
    
    def trigger_action(self, predicted_label):
        """Trigger action based on predicted gesture."""
        if predicted_label not in self.cfg.ACTION_MAP:
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if (current_time - self.last_trigger) < self.cfg.COOLDOWN_SECONDS:
            return False
        
        # Get key to press from ACTION_MAP
        key_to_press = self.cfg.ACTION_MAP[predicted_label]
        
        # Trigger action by pressing the arrow key
        try:
            pyautogui.press(key_to_press)
            self.last_trigger = current_time
            
            if self.cfg.PRINT_ACTIONS:
                arrow_symbol = "→" if key_to_press == "right" else "←"
                action = "Next" if predicted_label == "next" else "Previous"
                print(f"✓ {action} Slide: Pressed {arrow_symbol} arrow key")
            
            return True
        except Exception as e:
            print(f"Error triggering action: {e}")
            return False
    
    def run(self):
        """Main loop for real-time gesture recognition."""
        print("\n" + "="*60)
        print(f"{self.cfg.WINDOW_NAME}")
        print("="*60)
        print(f"Controls:")
        print(f"  Next Slide: Right Arrow Key (→)")
        print(f"  Previous Slide: Left Arrow Key (←)")
        print(f"  Pause/Resume: Press '{self.cfg.KEY_PAUSE}'")
        print(f"  Debug Mode: Press '{self.cfg.KEY_DEBUG}'")
        print(f"  Tracking Toggle: Press '{self.cfg.KEY_TRACKING}'")
        print(f"  Quit: Press '{self.cfg.KEY_QUIT}'")
        print("="*60)
        print(f"Model classes detected: {self.model_classes}")
        print(f"Label mapping: {self.cfg.LABEL_MAP}")
        print(f"Action mapping: {self.cfg.ACTION_MAP}")
        print(f"Hand Tracking: {'Enabled' if self.cfg.HAND_TRACKING_ENABLED else 'Disabled'}")
        print("="*60)
        print("Make sure your presentation software is active!")
        print("="*60)
        print("TIP: Show a new hand to switch control to it!")
        print("="*60)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.cfg.CAMERA_ID)
        if not self.cap.isOpened():
            print("❌ Unable to open webcam.")
            return
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.CAMERA_HEIGHT)
        
        # Initialize hand detector
        self.init_hand_detector()
        
        # Create window
        cv2.namedWindow(self.cfg.WINDOW_NAME)
        
        # FPS calculation
        frame_count = 0
        start_time = time.time()
        
        print("\nStarting gesture controller...")
        print("Show your hand to begin detection!\n")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to capture frame.")
                    break
                
                # Flip frame if configured
                if self.cfg.FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)
                
                # Predict gesture (skip if paused)
                if not self.paused:
                    predicted_label, confidence, display_frame = self.predict_gesture(frame)
                    
                    # Trigger action if consensus reached
                    if predicted_label:
                        # Already mapped in predict_gesture, use directly
                        if self.trigger_action(predicted_label):
                            # Flash ROI to indicate action
                            if self.current_roi:
                                x0, y0, x1, y1 = self.current_roi
                                cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 6)
                else:
                    # Just display frame when paused
                    display_frame = frame.copy()
                    cv2.putText(
                        display_frame,
                        "PAUSED - Press 'p' to resume",
                        (frame.shape[1]//2 - 150, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    self.fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Display frame
                cv2.imshow(self.cfg.WINDOW_NAME, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord(self.cfg.KEY_QUIT):
                    print("\nExiting...")
                    break
                elif key == ord(self.cfg.KEY_PAUSE):
                    self.paused = not self.paused
                    status = "PAUSED" if self.paused else "RESUMED"
                    print(f"\n{status} gesture detection")
                elif key == ord(self.cfg.KEY_DEBUG):
                    self.debug_mode = not self.debug_mode
                    status = "ENABLED" if self.debug_mode else "DISABLED"
                    print(f"\nDebug mode {status}")
                elif key == ord(self.cfg.KEY_TRACKING):
                    self.cfg.HAND_TRACKING_ENABLED = not self.cfg.HAND_TRACKING_ENABLED
                    if not self.cfg.HAND_TRACKING_ENABLED:
                        self.hand_tracker.reset()
                    status = "ENABLED" if self.cfg.HAND_TRACKING_ENABLED else "DISABLED"
                    print(f"\nHand tracking {status}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        
        if hasattr(self, 'hand_detector') and self.hand_detector and hasattr(self.hand_detector, 'close'):
            self.hand_detector.close()
             
        cv2.destroyAllWindows()
        
        if self.cfg.VERBOSE:
            print("Camera released. Goodbye!")


###############################################################################
# SIMPLE RUNNER
###############################################################################

def run_simple():
    """Run the gesture controller with default settings."""
    controller = GestureController()
    controller.run()


if __name__ == "__main__":
    run_simple()