#!/usr/bin/env python3
"""
Demo Script for Cricket Video Analysis
Shows how to use the cricket analyzer for detecting and analyzing cricket actions
"""

import cv2
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.yolo_detector import YOLODetector


def analyze_video(video_path: str, output_path: str = None):
    """
    Analyze a cricket video and optionally save annotated output
    
    Args:
        video_path: Path to input video
        output_path: Path to save annotated video (optional)
    """
    print(f"Analyzing video: {video_path}")
    
    # Initialize detector
    print("Initializing YOLO detector...")
    detector = YOLODetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    frame_count = 0
    detections_summary = []
    
    print("\nProcessing video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect objects
        detections = detector.detect(frame)
        detections_summary.append(len(detections))
        
        # Visualize detections
        annotated_frame = detector.visualize(frame, detections)
        
        # Write to output if enabled
        if writer:
            writer.write(annotated_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} ({progress:.1f}%) - Detected: {len(detections)} objects")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    # Print summary
    print("\n" + "="*50)
    print("Analysis Complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average detections per frame: {sum(detections_summary)/len(detections_summary):.2f}")
    print("="*50)


def analyze_webcam():
    """
    Real-time analysis from webcam
    Press 'q' to quit
    """
    print("Starting webcam analysis...")
    print("Press 'q' to quit")
    
    # Initialize detector
    detector = YOLODetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and visualize
        detections = detector.detect(frame)
        annotated_frame = detector.visualize(frame, detections)
        
        # Show frame
        cv2.imshow('Cricket Analysis - Press Q to quit', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam analysis stopped")


if __name__ == "__main__":
    print("="*50)
    print("Cricket Video Analysis System - Demo")
    print("="*50)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python demo.py <video_path> [output_path]")
        print("  python demo.py webcam")
        print()
        print("Examples:")
        print("  python demo.py cricket_match.mp4")
        print("  python demo.py cricket_match.mp4 output.mp4")
        print("  python demo.py webcam")
        sys.exit(1)
    
    if sys.argv[1].lower() == 'webcam':
        analyze_webcam()
    else:
        video_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        
        analyze_video(video_path, output_path)
