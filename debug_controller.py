"""
Debug Controller
A verbose version of the gesture controller for debugging and development.
Shows detailed information about detection, features, and predictions.

Usage:
    python debug_controller.py

Controls:
    Q - Quit
    D - Toggle debug overlay
    S - Save current frame
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
from pathlib import Path
from collections import deque
from skimage.feature import hog
import time

class DebugGestureController:
    """Debug version with verbose output and visualization."""
    
    def __init__(self):
        # Load model
        model_path = Path(__file__).parent / "artifacts" / "gesture_svm_v3.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        bundle = joblib.load(model_path)
        self.model = bundle['model']
        self.hog_params = bundle['hog_params']
        self.target_size = bundle['target_size']
        self.label_map = bundle['label_map']
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        print(f"✓ Model loaded: {model_path}")
        print(f"  Target size: {self.target_size}")
        print(f"  HOG params: {self.hog_params}")
        print(f"  Labels: {self.label_map}")
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Canny parameters (same as training)
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        
        # Stability
        self.SMOOTHING_FRAMES = 5
        self.CONFIDENCE_THRESHOLD = 0.70
        self.COOLDOWN_SECONDS = 2.0
        self.ROI_PADDING = 0.08
        
        self.prediction_history = deque(maxlen=self.SMOOTHING_FRAMES)
        self.last_action_time = 0
        
        # Debug state
        self.show_debug = True
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
    def extract_features(self, roi_gray):
        """Extract Canny + HOG features (same as training)."""
        # Resize
        resized = cv2.resize(roi_gray, self.target_size)
        
        # Canny edge detection
        edges = cv2.Canny(resized, self.canny_threshold1, self.canny_threshold2)
        edges_normalized = edges.astype(np.float32) / 255.0
        
        # HOG on edges
        hog_features = hog(edges_normalized, **self.hog_params)
        
        # Flatten canny and concatenate
        canny_flat = edges.flatten().astype(np.float32) / 255.0
        combined = np.concatenate([hog_features, canny_flat])
        
        return combined, edges, resized
    
    def get_hand_roi(self, frame, hand_landmarks):
        """Extract hand ROI with padding."""
        h, w = frame.shape[:2]
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        pad_x = (x_max - x_min) * self.ROI_PADDING
        pad_y = (y_max - y_min) * self.ROI_PADDING
        
        x0 = max(0, int(x_min - pad_x))
        y0 = max(0, int(y_min - pad_y))
        x1 = min(w, int(x_max + pad_x))
        y1 = min(h, int(y_max + pad_y))
        
        return (x0, y0, x1, y1)
    
    def process_frame(self, frame):
        """Process a single frame with debug info."""
        start_time = time.perf_counter()
        
        debug_info = {
            'hand_detected': False,
            'roi': None,
            'prediction': None,
            'confidence': 0,
            'probabilities': None,
            'consensus': None,
            'action': None,
            'edges': None,
            'resized': None
        }
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            debug_info['hand_detected'] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Get ROI
            roi = self.get_hand_roi(frame, hand_landmarks)
            debug_info['roi'] = roi
            x0, y0, x1, y1 = roi
            
            # Draw ROI box
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # Extract ROI and convert to grayscale
            roi_frame = frame[y0:y1, x0:x1]
            if roi_frame.size > 0:
                roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                
                # Extract features
                features, edges, resized = self.extract_features(roi_gray)
                debug_info['edges'] = edges
                debug_info['resized'] = resized
                
                # Predict
                pred = self.model.predict([features])[0]
                proba = self.model.predict_proba([features])[0]
                
                debug_info['prediction'] = pred
                debug_info['probabilities'] = dict(zip(self.model.classes_, proba))
                debug_info['confidence'] = max(proba)
                
                # Update history
                if debug_info['confidence'] >= self.CONFIDENCE_THRESHOLD:
                    self.prediction_history.append(pred)
                
                # Check consensus
                if len(self.prediction_history) == self.SMOOTHING_FRAMES:
                    if all(p == self.prediction_history[0] for p in self.prediction_history):
                        debug_info['consensus'] = self.prediction_history[0]
                        
                        # Check cooldown
                        current_time = time.time()
                        if current_time - self.last_action_time > self.COOLDOWN_SECONDS:
                            debug_info['action'] = debug_info['consensus']
                            self.last_action_time = current_time
                            self.prediction_history.clear()
        
        # Calculate FPS
        elapsed = time.perf_counter() - start_time
        self.fps_history.append(1 / elapsed if elapsed > 0 else 0)
        debug_info['fps'] = np.mean(self.fps_history)
        debug_info['inference_ms'] = elapsed * 1000
        
        self.frame_count += 1
        
        return frame, debug_info
    
    def draw_debug_overlay(self, frame, debug_info):
        """Draw debug information on frame."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        y_offset = 30
        line_height = 22
        
        # Draw info
        lines = [
            f"FPS: {debug_info.get('fps', 0):.1f}",
            f"Inference: {debug_info.get('inference_ms', 0):.1f} ms",
            f"Hand: {'YES' if debug_info['hand_detected'] else 'NO'}",
            f"Prediction: {debug_info['prediction'] or 'N/A'}",
            f"Confidence: {debug_info['confidence']*100:.1f}%",
            f"Consensus: {debug_info['consensus'] or 'N/A'}",
            f"Buffer: {len(self.prediction_history)}/{self.SMOOTHING_FRAMES}",
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, y_offset + i * line_height), 
                       font, 0.5, color, 1, cv2.LINE_AA)
        
        # Draw probabilities bar
        if debug_info['probabilities']:
            bar_y = 220
            bar_width = 150
            for label, prob in debug_info['probabilities'].items():
                cv2.putText(frame, f"{label}:", (20, bar_y), font, 0.5, color, 1)
                cv2.rectangle(frame, (100, bar_y - 12), (100 + int(bar_width * prob), bar_y), 
                             (0, 255, 0) if label == debug_info['prediction'] else (100, 100, 100), -1)
                cv2.putText(frame, f"{prob*100:.0f}%", (260, bar_y), font, 0.4, color, 1)
                bar_y += 25
        
        # Draw edge detection preview
        if debug_info['edges'] is not None and self.show_debug:
            edges_colored = cv2.cvtColor(debug_info['edges'], cv2.COLOR_GRAY2BGR)
            edges_small = cv2.resize(edges_colored, (128, 128))
            frame[10:138, w-138:w-10] = edges_small
            cv2.putText(frame, "Canny", (w-130, 155), font, 0.5, (255, 255, 0), 1)
        
        # Action indicator
        if debug_info['action']:
            action_text = f"ACTION: {debug_info['action'].upper()}"
            cv2.putText(frame, action_text, (w//2 - 100, h - 30), 
                       font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        
        return frame
    
    def run(self):
        """Main loop."""
        print("\n" + "="*60)
        print("DEBUG GESTURE CONTROLLER")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit")
        print("  D - Toggle debug overlay")
        print("  S - Save current frame")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror
            
            # Process
            frame, debug_info = self.process_frame(frame)
            
            # Draw overlay
            if self.show_debug:
                frame = self.draw_debug_overlay(frame, debug_info)
            
            cv2.imshow("Debug Gesture Controller", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
            elif key == ord('s'):
                filename = f"debug_frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nDebug controller stopped.")


if __name__ == "__main__":
    controller = DebugGestureController()
    controller.run()
