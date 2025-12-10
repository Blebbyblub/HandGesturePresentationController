import cv2
import numpy as np
import joblib
import time
import pyautogui
from skimage.feature import hog
from pathlib import Path
import argparse
from collections import deque
import statistics

###############################################################################
# CONFIGURATION SECTION
# Edit these values to customize the behavior
###############################################################################

class GestureConfig:
    """Configuration for Hand Gesture Presentation Controller"""
    
    # ========== MODEL SETTINGS ==========
    MODEL_PATH = "artifacts/gesture_svm_v2.pkl"  # Path to trained model
    BACKUP_MODEL_PATH = "artifacts/gesture_svm.pkl"  # Fallback model
    
    # ========== HOG PARAMETERS (from notebook) ==========
    HOG_PARAMS = {
        "orientations": 9,
        "pixels_per_cell": (16, 16),  # Changed from (8, 8) to (16, 16)
        "cells_per_block": (2, 2),
        "transform_sqrt": True,
        "block_norm": "L2-Hys",
        "feature_vector": True,
    }
    
    TARGET_IMAGE_SIZE = (128, 128)  # Image size for feature extraction
    
    # ========== GESTURE MAPPING ==========
    LABEL_MAP = {
        "next": "next",
        "back": "previous",
        "prev": "previous"
    }
    
    ACTION_MAP = {
        "next": "right",  # Right arrow key for next slide
        "previous": "left"  # Left arrow key for previous slide
    }
    
    # ========== INFERENCE SETTINGS ==========
    CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence to consider prediction
    COOLDOWN_SECONDS = 2.0  # Minimum time between triggers
    
    # ========== HAND DETECTION SETTINGS ==========
    USE_MEDIAPIPE = True  # Use MediaPipe for hand detection (recommended)
    AUTO_ROI_PADDING = 50  # Padding around detected hand (pixels)
    MIN_HAND_SIZE = 100  # Minimum hand size to detect (pixels)
    
    # ========== SMOOTHING SETTINGS ==========
    SMOOTHING_FRAMES = 5  # Number of frames for consensus voting
    CONSENSUS_REQUIRED = "all"  # "all" = all frames must agree, "majority" = majority vote
    
    # ========== DISPLAY SETTINGS ==========
    DISPLAY_FPS = True  # Show FPS counter
    SHOW_HISTORY = True  # Show prediction history
    SHOW_CONFIDENCE = True  # Show confidence score
    SHOW_INSTRUCTIONS = False  # Show control instructions (only 'q' to quit)
    
    # ========== VISUAL APPEARANCE ==========
    ROI_COLOR_HIGH_CONF = (0, 255, 0)  # Green for high confidence
    ROI_COLOR_LOW_CONF = (0, 165, 255)  # Orange for low confidence
    TEXT_COLOR = (255, 255, 255)  # White text
    WINDOW_NAME = "Hand Gesture Controller V2"
    
    # ========== CAMERA SETTINGS ==========
    CAMERA_ID = 0  # Default camera ID
    CAMERA_WIDTH = 640  # Desired camera width
    CAMERA_HEIGHT = 480  # Desired camera height
    FLIP_HORIZONTAL = True  # Flip camera horizontally for mirror effect
    
    # ========== CONTROL KEYS ==========
    KEY_QUIT = 'q'  # Only key needed
    
    # ========== DEBUG SETTINGS ==========
    VERBOSE = True  # Print debug information
    PRINT_PREDICTIONS = False  # Print every prediction (can be noisy)
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*50)
        print("GESTURE CONTROLLER CONFIGURATION")
        print("="*50)
        
        sections = [
            ("Model Settings", [
                f"Model Path: {cls.MODEL_PATH}",
                f"Backup Model: {cls.BACKUP_MODEL_PATH}",
                f"Confidence Threshold: {cls.CONFIDENCE_THRESHOLD}",
                f"Cooldown: {cls.COOLDOWN_SECONDS}s",
            ]),
            ("HOG Parameters", [
                f"Image Size: {cls.TARGET_IMAGE_SIZE}",
                f"Pixels per Cell: {cls.HOG_PARAMS['pixels_per_cell']}",
                f"Orientations: {cls.HOG_PARAMS['orientations']}",
            ]),
            ("Hand Detection", [
                f"Use MediaPipe: {cls.USE_MEDIAPIPE}",
                f"ROI Padding: {cls.AUTO_ROI_PADDING}px",
                f"Min Hand Size: {cls.MIN_HAND_SIZE}px",
            ]),
            ("Smoothing", [
                f"Smoothing Frames: {cls.SMOOTHING_FRAMES}",
                f"Consensus Required: {cls.CONSENSUS_REQUIRED}",
            ]),
            ("Display", [
                f"Show FPS: {cls.DISPLAY_FPS}",
                f"Show History: {cls.SHOW_HISTORY}",
                f"Window Name: {cls.WINDOW_NAME}",
            ]),
            ("Camera", [
                f"Camera ID: {cls.CAMERA_ID}",
                f"Resolution: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}",
                f"Flip Horizontal: {cls.FLIP_HORIZONTAL}",
            ]),
        ]
        
        for section_name, settings in sections:
            print(f"\n{section_name}:")
            for setting in settings:
                print(f"  {setting}")
        
        print("="*50)


###############################################################################
# MAIN GESTURE CONTROLLER CLASS
###############################################################################

class GestureController:
    def __init__(self, config_obj=None):
        """
        Initialize the Gesture Controller for real-time hand gesture recognition.
        
        Args:
            config_obj: Configuration object (defaults to GestureConfig)
        """
        # Use provided config or default
        self.cfg = config_obj if config_obj else GestureConfig
        
        # Print configuration
        if self.cfg.VERBOSE:
            self.cfg.print_config()
        
        # Initialize state variables
        self.hand_detector = None
        self.current_roi = None
        self.last_good_roi = None
        self.hand_detected = False
        self.last_trigger = 0.0
        self.cap = None
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=self.cfg.SMOOTHING_FRAMES)
        
        # Load the trained model
        self.load_model()
        
    def load_model(self):
        """Load the trained SVM model and preprocessing parameters."""
        model_path = Path(self.cfg.MODEL_PATH)
        
        try:
            bundle = joblib.load(model_path)
            print(f"✓ Model loaded successfully from {model_path}")
        except FileNotFoundError:
            if self.cfg.VERBOSE:
                print(f"⚠️ V2 model not found. Trying backup model...")
            # Try backup model
            backup_path = Path(self.cfg.BACKUP_MODEL_PATH)
            try:
                bundle = joblib.load(backup_path)
                print(f"✓ Backup model loaded from {backup_path}")
            except FileNotFoundError:
                print(f"❌ Error: No model found at {model_path} or {backup_path}")
                print("Please run the training notebook first.")
                raise
        
        # Extract model components
        self.clf = bundle["model"]
        self.target_size = tuple(bundle.get("target_size", self.cfg.TARGET_IMAGE_SIZE))
        
        # Use HOG parameters from config (overriding saved ones if needed)
        self.hog_params = self.cfg.HOG_PARAMS.copy()
        
        if self.cfg.VERBOSE:
            print(f"✓ Target image size: {self.target_size}")
            print(f"✓ Available classes: {self.clf.classes_}")
            print(f"✓ Model kernel: {self.clf.kernel}")
    
    def init_hand_detector(self):
        """Initialize hand detector based on configuration."""
        if self.cfg.USE_MEDIAPIPE:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hand_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                if self.cfg.VERBOSE:
                    print("✓ MediaPipe hand detector initialized")
                return True
            except ImportError:
                if self.cfg.VERBOSE:
                    print("⚠️ MediaPipe not installed. Using skin detection.")
                    print("   Install with: pip install mediapipe")
                return False
        
        # If MediaPipe is disabled or not available
        if self.cfg.VERBOSE:
            print("✓ Using OpenCV skin detection")
        return True
    
    def detect_hand_with_mediapipe(self, frame):
        """Detect hand using MediaPipe."""
        if not self.hand_detector:
            return None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detector.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box from landmarks
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Add padding and ensure minimum size
            roi = self._adjust_roi_with_constraints(x_min, y_min, x_max, y_max, frame.shape)
            
            if roi:
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
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Skip if too small
            if w < 50 or h < 50:
                self.hand_detected = False
                return None
            
            # Add padding and adjust
            x_min = max(0, x - self.cfg.AUTO_ROI_PADDING)
            y_min = max(0, y - self.cfg.AUTO_ROI_PADDING)
            x_max = min(frame.shape[1], x + w + self.cfg.AUTO_ROI_PADDING)
            y_max = min(frame.shape[0], y + h + self.cfg.AUTO_ROI_PADDING)
            
            roi = (x_min, y_min, x_max, y_max)
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
        """Get ROI by detecting hand position."""
        if self.hand_detector and self.cfg.USE_MEDIAPIPE:
            roi = self.detect_hand_with_mediapipe(frame)
        else:
            roi = self.detect_hand_with_skincolor(frame)
        
        self.current_roi = roi
        return roi
    
    def preprocess_frame(self, frame, roi_bounds):
        """Preprocess the ROI frame for inference."""
        if roi_bounds is None:
            return None
        
        x0, y0, x1, y1 = roi_bounds
        
        # Validate ROI
        if x1 <= x0 or y1 <= y0:
            return None
        
        # Extract and preprocess ROI
        roi_frame = frame[y0:y1, x0:x1]
        if roi_frame.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize
        resized = cv2.resize(gray, self.target_size)
        
        # Normalize
        normalized = resized.astype("float32") / 255.0
        
        # Extract HOG features
        descriptor = hog(normalized, **self.hog_params)
        
        return descriptor.reshape(1, -1)
    
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
    
    def predict_gesture(self, frame):
        """Predict gesture from frame with smoothing."""
        display_frame = frame.copy()
        
        # Get ROI
        current_roi = self.get_current_roi(frame)
        
        # Initialize values
        current_label = None
        confidence = 0.0
        
        if current_roi and self.hand_detected:
            # Extract features and predict
            descriptor = self.preprocess_frame(frame, current_roi)
            
            if descriptor is not None:
                proba = self.clf.predict_proba(descriptor)[0]
                top_idx = int(np.argmax(proba))
                current_label = self.clf.classes_[top_idx]
                confidence = float(proba[top_idx])
                
                if self.cfg.PRINT_PREDICTIONS:
                    print(f"Prediction: {current_label} ({confidence:.2f})")
            
            # Update history
            if confidence > self.cfg.CONFIDENCE_THRESHOLD:
                self.prediction_history.append(current_label)
            else:
                self.prediction_history.append("neutral")
            
            # Draw ROI and info
            self._draw_roi_and_info(display_frame, current_roi, current_label, confidence)
        else:
            # No hand detected
            self.prediction_history.clear()
            self._draw_no_hand_message(display_frame)
        
        # Get consensus prediction
        consensus_label = self.get_consensus_prediction()
        
        # Draw UI elements
        self._draw_ui_elements(display_frame, consensus_label)
        
        return consensus_label, confidence, display_frame
    
    def _draw_roi_and_info(self, frame, roi, label, confidence):
        """Draw ROI and prediction info on frame."""
        x0, y0, x1, y1 = roi
        
        # Choose color based on confidence
        if confidence >= self.cfg.CONFIDENCE_THRESHOLD:
            color = self.cfg.ROI_COLOR_HIGH_CONF
        else:
            color = self.cfg.ROI_COLOR_LOW_CONF
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Draw ROI border
        thickness = 3 if confidence >= self.cfg.CONFIDENCE_THRESHOLD else 2
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
        
        # Draw label and confidence
        if self.cfg.SHOW_CONFIDENCE:
            label_text = f"{label}: {confidence:.2f}"
        else:
            label_text = f"{label}"
        
        cv2.putText(
            frame,
            label_text,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    
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
        if consensus_label:
            cv2.putText(
                frame,
                f"Consensus: {consensus_label} ✓",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
            y_offset += line_height
        
        # Simple quit instruction only
        if self.cfg.SHOW_INSTRUCTIONS:
            cv2.putText(
                frame,
                f"Press '{self.cfg.KEY_QUIT}' to quit",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.cfg.TEXT_COLOR,
                1,
            )
            y_offset += line_height
        
        # FPS
        if self.cfg.DISPLAY_FPS and hasattr(self, 'fps'):
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (10, frame.shape[0] - 10),
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
        
        # Get key to press
        key_to_press = self.cfg.ACTION_MAP[predicted_label]
        
        # Trigger action
        try:
            pyautogui.press(key_to_press)
            self.last_trigger = current_time
            
            if self.cfg.VERBOSE:
                action = "Next" if predicted_label == "next" else "Previous"
                print(f"✓ Triggered: {action} Slide")
            
            return True
        except Exception as e:
            if self.cfg.VERBOSE:
                print(f"Error triggering action: {e}")
            return False
    
    def run(self):
        """Main loop for real-time gesture recognition."""
        print("\n" + "="*50)
        print(f"{self.cfg.WINDOW_NAME}")
        print("="*50)
        print(f"Controls:")
        print(f"  {self.cfg.KEY_QUIT}: Quit")
        print("="*50)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.cfg.CAMERA_ID)
        if not self.cap.isOpened():
            print("❌ Error: Unable to open webcam.")
            return
        
        # Set camera resolution if possible
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
                    print("❌ Error: Failed to capture frame.")
                    break
                
                # Flip frame if configured
                if self.cfg.FLIP_HORIZONTAL:
                    frame = cv2.flip(frame, 1)
                
                # Predict gesture
                predicted_label, confidence, display_frame = self.predict_gesture(frame)
                
                # Trigger action if consensus reached
                if predicted_label:
                    if self.trigger_action(predicted_label):
                        # Flash ROI to indicate action
                        if self.current_roi:
                            x0, y0, x1, y1 = self.current_roi
                            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 6)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    self.fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Display frame
                cv2.imshow(self.cfg.WINDOW_NAME, display_frame)
                
                # Handle keyboard input - only 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord(self.cfg.KEY_QUIT):
                    print("\nExiting gesture controller...")
                    break
                
        except KeyboardInterrupt:
            print("\nGesture controller interrupted by user.")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        if self.hand_detector:
            self.hand_detector.close()
        cv2.destroyAllWindows()
        
        if self.cfg.VERBOSE:
            print("Camera released. Goodbye!")


###############################################################################
# SIMPLE RUNNER (no command line arguments needed)
###############################################################################

def run_simple():
    """Run the gesture controller with default settings."""
    controller = GestureController(GestureConfig)
    controller.run()


def run_with_custom_config():
    """Run with custom configuration (edit values below)."""
    
    # Create custom configuration by modifying the default
    class CustomConfig(GestureConfig):
        # Override any settings you want
        CONFIDENCE_THRESHOLD = 0.65  # Lower threshold
        SMOOTHING_FRAMES = 3  # Fewer frames for faster response
        AUTO_ROI_PADDING = 40  # Less padding
        USE_MEDIAPIPE = False  # Use skin detection instead
        SHOW_INSTRUCTIONS = False  # Don't show instructions on screen
    
    # Run with custom config
    controller = GestureController(CustomConfig)
    controller.run()


if __name__ == "__main__":
    # Option 1: Run with default settings (recommended)
    run_simple()
    
    # Option 2: Run with custom configuration (edit above)
    # run_with_custom_config()