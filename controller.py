import cv2
import numpy as np
import joblib
import time
import pyautogui
from skimage.feature import hog
from pathlib import Path
import argparse

class GestureController:
    def __init__(self, model_path=None, confidence_threshold=0.65, 
                 cooldown_seconds=2.0, use_mediapipe=True):
        """
        Initialize the Gesture Controller for real-time hand gesture recognition.
        
        Args:
            model_path: Path to the trained model bundle (.pkl file)
            confidence_threshold: Minimum confidence to trigger a gesture
            cooldown_seconds: Minimum time between consecutive triggers
            use_mediapipe: Use MediaPipe for hand detection (recommended)
        """
        # Set default model path if not provided
        if model_path is None:
            base_dir = Path.cwd()
            model_path = base_dir / "artifacts" / "gesture_svm.pkl"
        
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.use_mediapipe = use_mediapipe
        
        # For auto hand detection
        self.hand_detector = None
        self.auto_roi_padding = 50  # Padding around detected hand
        self.current_roi = None
        self.last_good_roi = None
        self.hand_detected = False  # Track if hand is currently detected
        
        # Load the trained model and parameters
        self.load_model()
        
        # Initialize state variables
        self.last_trigger = 0.0
        self.cap = None
        
    def load_model(self):
        """Load the trained SVM model and preprocessing parameters."""
        try:
            bundle = joblib.load(self.model_path)
            self.clf = bundle["model"]
            self.hog_params = bundle["hog_params"]
            self.target_size = tuple(bundle["target_size"])
            self.label_map = bundle.get("label_map", {})
            
            print(f"✓ Model loaded successfully from {self.model_path}")
            print(f"✓ Target image size: {self.target_size}")
            print(f"✓ Available classes: {self.clf.classes_}")
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please run the training notebook first or provide the correct path.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def init_hand_detector(self):
        """Initialize hand detector."""
        if self.use_mediapipe:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hand_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✓ MediaPipe hand detector initialized")
                return True
            except ImportError:
                print("⚠️ MediaPipe not installed. Using skin detection.")
                print("   Install with: pip install mediapipe")
                self.use_mediapipe = False
        
        # Initialize OpenCV-based skin detector (fallback)
        print("✓ Using OpenCV skin detection")
        return True
    
    def detect_hand_with_mediapipe(self, frame):
        """Detect hand using MediaPipe and return ROI around it."""
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
            
            # Add padding
            x_min = max(0, x_min - self.auto_roi_padding)
            y_min = max(0, y_min - self.auto_roi_padding)
            x_max = min(w, x_max + self.auto_roi_padding)
            y_max = min(h, y_max + self.auto_roi_padding)
            
            # Ensure minimum size (at least 100x100 pixels)
            if (x_max - x_min) < 100:
                center_x = (x_min + x_max) // 2
                x_min = max(0, center_x - 50)
                x_max = min(w, center_x + 50)
            
            if (y_max - y_min) < 100:
                center_y = (y_min + y_max) // 2
                y_min = max(0, center_y - 50)
                y_max = min(h, center_y + 50)
            
            # Update last good ROI
            self.last_good_roi = (x_min, y_min, x_max, y_max)
            self.hand_detected = True
            return self.last_good_roi
        else:
            self.hand_detected = False
            return None
    
    def detect_hand_with_skincolor(self, frame):
        """Detect hand using skin color detection (fallback method)."""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjust these values based on lighting)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create skin mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Skip if too small (likely noise)
            if w < 50 or h < 50:
                self.hand_detected = False
                return None
            
            # Add padding
            padding = self.auto_roi_padding
            x_min = max(0, x - padding)
            y_min = max(0, y - padding)
            x_max = min(frame.shape[1], x + w + padding)
            y_max = min(frame.shape[0], y + h + padding)
            
            # Update last good ROI
            self.last_good_roi = (x_min, y_min, x_max, y_max)
            self.hand_detected = True
            return self.last_good_roi
        else:
            self.hand_detected = False
            return None
    
    def get_current_roi(self, frame):
        """Get ROI by detecting hand position. Returns None if no hand."""
        # Try to detect hand
        if self.use_mediapipe and self.hand_detector:
            roi = self.detect_hand_with_mediapipe(frame)
        else:
            roi = self.detect_hand_with_skincolor(frame)
        
        if roi:
            self.current_roi = roi
            return roi
        else:
            # No hand detected
            self.current_roi = None
            return None
    
    def preprocess_frame(self, frame, roi_bounds):
        """
        Preprocess the ROI frame for inference.
        
        Args:
            frame: Input frame from webcam
            roi_bounds: Current ROI bounds (or None if no hand)
            
        Returns:
            HOG descriptor for the ROI, or None if no hand
        """
        if roi_bounds is None:
            # No hand detected
            return None
        
        x0, y0, x1, y1 = roi_bounds
        
        # Ensure ROI is valid
        if x1 <= x0 or y1 <= y0:
            return None
        
        # Extract ROI
        roi_frame = frame[y0:y1, x0:x1]
        
        # Check if ROI is empty
        if roi_frame.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        resized = cv2.resize(gray, self.target_size)
        
        # Normalize
        normalized = resized.astype("float32") / 255.0
        
        # Extract HOG features
        descriptor = hog(normalized, **self.hog_params)
        
        return descriptor.reshape(1, -1)
    
    def predict_gesture(self, frame):
        """
        Predict gesture from the ROI in the frame.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            predicted_label: The predicted gesture class or None if no hand
            confidence: Confidence score (0-1) or 0 if no hand
            processed_frame: Frame with ROI overlay
        """
        # Create a copy for display
        display_frame = frame.copy()
        
        # Get current ROI by detecting hand
        current_roi = self.get_current_roi(frame)
        
        # Initialize default values
        predicted_label = None
        confidence = 0.0
        
        if current_roi and self.hand_detected:
            # Extract and preprocess ROI
            descriptor = self.preprocess_frame(frame, current_roi)
            
            if descriptor is not None:
                # Predict
                proba = self.clf.predict_proba(descriptor)[0]
                top_idx = int(np.argmax(proba))
                predicted_label = self.clf.classes_[top_idx]
                confidence = float(proba[top_idx])
            
            # Draw ROI rectangle with color based on confidence
            x0, y0, x1, y1 = current_roi
            
            # Draw semi-transparent overlay
            overlay = display_frame.copy()
            if confidence >= self.confidence_threshold:
                color = (0, 255, 0)  # Green - high confidence
                overlay_color = (0, 255, 0)
            else:
                color = (0, 165, 255)  # Orange - low confidence
                overlay_color = (0, 165, 255)
                
            cv2.rectangle(overlay, (x0, y0), (x1, y1), overlay_color, -1)
            cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)
            
            # Draw ROI border
            thickness = 3 if confidence >= self.confidence_threshold else 2
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), color, thickness)
            
            # Draw label and confidence
            label_text = f"{predicted_label}: {confidence:.2f}"
            cv2.putText(
                display_frame,
                label_text,
                (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            # No hand detected - show message
            h, w = frame.shape[:2]
            cv2.putText(
                display_frame,
                "NO HAND DETECTED",
                (w//2 - 100, h//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            
            # Draw hand silhouette as hint
            cv2.putText(
                display_frame,
                "Show your hand to begin",
                (w//2 - 120, h//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )
        
        # Draw detection method
        method_text = "MediaPipe" if (self.use_mediapipe and self.hand_detector) else "Skin Detection"
        cv2.putText(
            display_frame,
            f"Hand Detection: {method_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        
        # Draw hand status
        status_color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        status_text = "HAND DETECTED" if self.hand_detected else "NO HAND"
        cv2.putText(
            display_frame,
            f"Status: {status_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            1,
        )
        
        # Draw instructions
        instructions = [
            "Move hand anywhere in frame",
            "Press 'q' to quit",
            "Press 'c' to change confidence",
            "Press '+' to increase padding",
            "Press '-' to decrease padding"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(
                display_frame,
                text,
                (10, 90 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        
        # Draw FPS (if available)
        if hasattr(self, 'fps'):
            cv2.putText(
                display_frame,
                f"FPS: {self.fps:.1f}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        
        return predicted_label, confidence, display_frame
    
    def trigger_action(self, predicted_label):
        """
        Trigger the appropriate action based on predicted gesture.
        
        Args:
            predicted_label: The predicted gesture class (None if no hand)
        """
        if predicted_label is None:
            return False
        
        current_time = time.time()
        
        # Check cooldown
        if (current_time - self.last_trigger) < self.cooldown_seconds:
            return False
        
        # Map gesture to action
        if predicted_label == "next":
            key_to_press = "right"
            action = "Next Slide"
        elif predicted_label == "previous":
            key_to_press = "left"
            action = "Previous Slide"
        else:
            return False
        
        # Trigger action
        try:
            pyautogui.press(key_to_press)
            self.last_trigger = current_time
            print(f"✓ Triggered: {action} ({predicted_label})")
            return True
        except Exception as e:
            print(f"Error triggering action: {e}")
            return False
    
    def run(self):
        """Main loop for real-time gesture recognition."""
        print("\n" + "="*50)
        print("HAND GESTURE PRESENTATION CONTROLLER")
        print("="*50)
        print("Features:")
        print("- Automatic hand tracking (no fixed box needed)")
        print("- Hand can move anywhere in the frame")
        print("- No detection when hand is not present")
        print(f"- Using: {'MediaPipe' if self.use_mediapipe else 'Skin Detection'}")
        print("="*50)
        print("Controls:")
        print("- Show hand: ROI automatically follows")
        print("- Remove hand: Detection stops immediately")
        print("- 'q': Quit")
        print("- 'c': Change confidence threshold")
        print("- '+': Increase padding around hand")
        print("- '-': Decrease padding around hand")
        print("="*50 + "\n")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Unable to open webcam.")
            print("Please ensure a camera is connected and not in use.")
            return
        
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        
        # Initialize hand detector
        if not self.init_hand_detector():
            print("Error: Failed to initialize hand detector.")
            return
        
        # Create window
        cv2.namedWindow("Hand Gesture Controller")
        
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Cooldown: {self.cooldown_seconds} seconds")
        print(f"ROI padding: {self.auto_roi_padding} pixels")
        print("\nStarting automatic gesture recognition...")
        print("Show your hand to begin detection!")
        
        # FPS calculation
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Flip frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Predict gesture (returns None if no hand)
                predicted_label, confidence, display_frame = self.predict_gesture(frame)
                
                # Trigger action only if hand is detected AND confidence is high enough
                if predicted_label is not None and confidence >= self.confidence_threshold:
                    self.trigger_action(predicted_label)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:  # Update FPS every second
                    self.fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Display frame
                cv2.imshow("Hand Gesture Controller", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting gesture controller...")
                    break
                elif key == ord('c'):
                    # Change confidence threshold
                    try:
                        new_threshold = float(input("\nEnter new confidence threshold (0.1-0.9): "))
                        if 0.1 <= new_threshold <= 0.9:
                            self.confidence_threshold = new_threshold
                            print(f"Confidence threshold set to: {new_threshold}")
                        else:
                            print("Threshold must be between 0.1 and 0.9")
                    except ValueError:
                        print("Invalid input. Using current threshold.")
                    
                elif key == ord('+'):
                    # Increase ROI padding
                    self.auto_roi_padding = min(150, self.auto_roi_padding + 10)
                    print(f"\nPadding increased to: {self.auto_roi_padding} pixels")
                    
                elif key == ord('-'):
                    # Decrease ROI padding
                    self.auto_roi_padding = max(20, self.auto_roi_padding - 10)
                    print(f"\nPadding decreased to: {self.auto_roi_padding} pixels")
                    
                elif key == ord('d'):
                    # Toggle detection method (if MediaPipe is available)
                    if self.use_mediapipe:
                        self.use_mediapipe = not self.use_mediapipe
                        method = "MediaPipe" if self.use_mediapipe else "Skin Detection"
                        print(f"\nDetection method changed to: {method}")
                
        except KeyboardInterrupt:
            print("\nGesture controller interrupted by user.")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            if self.hand_detector:
                self.hand_detector.close()
            cv2.destroyAllWindows()
            print("Camera released. Goodbye!")

def main():
    """Main function to run the gesture controller."""
    parser = argparse.ArgumentParser(description='Hand Gesture Presentation Controller with Auto Tracking')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (default: artifacts/gesture_svm.pkl)')
    parser.add_argument('--confidence', type=float, default=0.65,
                       help='Confidence threshold (default: 0.65)')
    parser.add_argument('--cooldown', type=float, default=2.0,
                       help='Cooldown between actions in seconds (default: 2.0)')
    parser.add_argument('--padding', type=int, default=50,
                       help='Padding around detected hand in pixels (default: 50)')
    parser.add_argument('--no-mediapipe', action='store_true',
                       help='Disable MediaPipe (use skin detection instead)')
    
    args = parser.parse_args()
    
    # Create gesture controller
    controller = GestureController(
        model_path=args.model,
        confidence_threshold=args.confidence,
        cooldown_seconds=args.cooldown,
        use_mediapipe=not args.no_mediapipe
    )
    
    # Set padding
    controller.auto_roi_padding = args.padding
    
    # Run the controller
    controller.run()

if __name__ == "__main__":
    main()