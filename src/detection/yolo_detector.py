"""
YOLO-based Object Detector for Cricket Analysis
Detects players, bat, ball, and wickets in cricket videos
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import os


class YOLODetector:
    """
    YOLOv8-based detector for cricket objects
    """
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained YOLO model (None for default)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pretrained YOLO model
            self.model = YOLO('yolov8n.pt')  # Nano model for fast inference
            print("Using pretrained YOLOv8n model")
        
        # Cricket-specific class names
        self.cricket_classes = ['player', 'bat', 'ball', 'wicket', 'umpire']
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of detections with bounding boxes and confidence
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.model.names[cls]
                }
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in multiple frames (batch processing)
        
        Args:
            frames: List of input images
            
        Returns:
            List of detection results for each frame
        """
        results = self.model(frames, conf=self.conf_threshold, verbose=False)
        
        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': self.model.names[cls]
                }
                frame_detections.append(detection)
            all_detections.append(frame_detections)
        
        return all_detections
    
    def visualize(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = f"{det['class_name']}: {conf:.2f}"
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output


if __name__ == "__main__":
    # Test the detector
    detector = YOLODetector()
    print("YOLODetector initialized successfully!")
    print(f"Model classes: {detector.model.names}")
