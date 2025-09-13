"""
Visualization utilities for object detection system
Handles drawing bounding boxes, labels, confidence scores, and performance metrics
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import colorsys
from PIL import Image, ImageDraw, ImageFont
import io
import base64

class DetectionVisualizer:
    """Main class for visualizing object detection results"""
    
    def __init__(self, class_names: List[str], confidence_threshold: float = 0.5):
        """
        Initialize the visualizer
        
        Args:
            class_names: List of class names for the detection model
            confidence_threshold: Minimum confidence to display detections
        """
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.colors = self._generate_colors(len(class_names))
        
        # Font settings for text rendering
        self.font_scale = 0.6
        self.font_thickness = 2
        self.text_offset = 5
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        colors = []
        for i in range(num_classes):
            # Use HSV color space for better color distribution
            hue = i / num_classes
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to 0-255 range and BGR for OpenCV
            colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))
        return colors
    
    def draw_detections(self, 
                       image: np.ndarray,
                       detections: Dict,
                       show_confidence: bool = True,
                       show_class_names: bool = True,
                       line_thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            detections: Dictionary with 'boxes', 'scores', 'classes' keys
            show_confidence: Whether to display confidence scores
            show_class_names: Whether to display class names
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        boxes = detections.get('boxes', [])
        scores = detections.get('scores', [])
        classes = detections.get('classes', [])
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
            if score < self.confidence_threshold:
                continue
                
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for this class
            color = self.colors[int(class_id) % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label text
            label_parts = []
            if show_class_names and class_id < len(self.class_names):
                label_parts.append(self.class_names[int(class_id)])
            if show_confidence:
                label_parts.append(f"{score:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                )
                
                # Draw text background
                cv2.rectangle(result_image, 
                            (x1, y1 - text_height - baseline - self.text_offset),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Draw text
                cv2.putText(result_image, label,
                          (x1, y1 - self.text_offset),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          self.font_scale,
                          (255, 255, 255),
                          self.font_thickness)
        
        return result_image
    
    def create_detection_grid(self, 
                            images: List[np.ndarray],
                            detections_list: List[Dict],
                            titles: Optional[List[str]] = None,
                            grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a grid of images with detections
        
        Args:
            images: List of input images
            detections_list: List of detection dictionaries
            titles: Optional titles for each image
            grid_size: Optional (rows, cols) for grid layout
            
        Returns:
            Grid image with all detections
        """
        if not images:
            return np.array([])
            
        num_images = len(images)
        
        # Calculate grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = grid_size
        
        # Process each image
        processed_images = []
        for i, (img, dets) in enumerate(zip(images, detections_list)):
            processed_img = self.draw_detections(img, dets)
            
            # Add title if provided
            if titles and i < len(titles):
                processed_img = self._add_title(processed_img, titles[i])
                
            processed_images.append(processed_img)
        
        # Pad with empty images if needed
        while len(processed_images) < rows * cols:
            empty_img = np.zeros_like(processed_images[0])
            processed_images.append(empty_img)
        
        # Create grid
        grid_rows = []
        for row in range(rows):
            row_images = processed_images[row * cols:(row + 1) * cols]
            if row_images:
                grid_row = np.hstack(row_images)
                grid_rows.append(grid_row)
        
        if grid_rows:
            grid_image = np.vstack(grid_rows)
            return grid_image
        else:
            return np.array([])
    
    def _add_title(self, image: np.ndarray, title: str) -> np.ndarray:
        """Add title to image top"""
        height, width = image.shape[:2]
        title_height = 50
        
        # Create new image with space for title
        titled_image = np.zeros((height + title_height, width, 3), dtype=np.uint8)
        titled_image[title_height:, :] = image
        
        # Add title text
        cv2.putText(titled_image, title,
                   (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (255, 255, 255), 2)
        
        return titled_image

class MetricsVisualizer:
    """Class for visualizing training and evaluation metrics"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize metrics visualizer"""
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_training_curves(self,
                           train_losses: List[float],
                           val_losses: List[float],
                           train_metrics: Optional[Dict[str, List[float]]] = None,
                           val_metrics: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training curves for loss and metrics
        
        Args:
            train_losses: Training loss values
            val_losses: Validation loss values
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        # Determine number of subplots needed
        num_metrics = 1 + (len(train_metrics) if train_metrics else 0)
        fig, axes = plt.subplots(1, min(num_metrics, 3), figsize=(15, 5))
        
        if num_metrics == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot loss
        axes[0].plot(epochs, train_losses, label='Training Loss', 
                    color=self.colors[0], linewidth=2)
        axes[0].plot(epochs, val_losses, label='Validation Loss', 
                    color=self.colors[1], linewidth=2)
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot additional metrics
        if train_metrics and val_metrics and len(axes) > 1:
            metric_idx = 1
            for metric_name in list(train_metrics.keys())[:2]:  # Plot up to 2 additional metrics
                if metric_idx >= len(axes):
                    break
                    
                axes[metric_idx].plot(epochs, train_metrics[metric_name], 
                                    label=f'Training {metric_name}', 
                                    color=self.colors[2], linewidth=2)
                axes[metric_idx].plot(epochs, val_metrics[metric_name], 
                                    label=f'Validation {metric_name}', 
                                    color=self.colors[3], linewidth=2)
                axes[metric_idx].set_title(f'Training and Validation {metric_name}')
                axes[metric_idx].set_xlabel('Epoch')
                axes[metric_idx].set_ylabel(metric_name)
                axes[metric_idx].legend()
                axes[metric_idx].grid(True, alpha=0.3)
                metric_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            class_names: List[str],
                            normalize: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_pr_curve(self,
                     precision: np.ndarray,
                     recall: np.ndarray,
                     ap_score: float,
                     class_name: str = '',
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            ap_score: Average Precision score
            class_name: Name of the class
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, 
               label=f'{class_name} (AP = {ap_score:.3f})')
        ax.fill_between(recall, precision, alpha=0.3)
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(f'Precision-Recall Curve {class_name}', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_detection_summary(self,
                               detections_per_class: Dict[str, int],
                               confidence_scores: List[float],
                               processing_times: List[float],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create summary visualization with multiple metrics
        
        Args:
            detections_per_class: Count of detections per class
            confidence_scores: List of all confidence scores
            processing_times: List of processing times
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution bar plot
        if detections_per_class:
            classes = list(detections_per_class.keys())
            counts = list(detections_per_class.values())
            
            axes[0, 0].bar(classes, counts, color=self.colors[0], alpha=0.7)
            axes[0, 0].set_title('Detections per Class')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence score histogram
        if confidence_scores:
            axes[0, 1].hist(confidence_scores, bins=30, color=self.colors[1], 
                          alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Confidence Score Distribution')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(confidence_scores), color='red', 
                             linestyle='--', label=f'Mean: {np.mean(confidence_scores):.3f}')
            axes[0, 1].legend()
        
        # Processing time plot
        if processing_times:
            axes[1, 0].plot(processing_times, color=self.colors[2], linewidth=1)
            axes[1, 0].set_title('Processing Time per Frame')
            axes[1, 0].set_xlabel('Frame Number')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].axhline(np.mean(processing_times), color='red', 
                             linestyle='--', label=f'Mean: {np.mean(processing_times):.3f}s')
            axes[1, 0].legend()
        
        # Performance summary text
        axes[1, 1].axis('off')
        summary_text = f"""
        Performance Summary:
        
        Total Detections: {sum(detections_per_class.values()) if detections_per_class else 0}
        Average Confidence: {np.mean(confidence_scores):.3f} ± {np.std(confidence_scores):.3f}
        Average Processing Time: {np.mean(processing_times):.3f}s
        FPS: {1/np.mean(processing_times):.1f}
        
        Min Processing Time: {np.min(processing_times):.3f}s
        Max Processing Time: {np.max(processing_times):.3f}s
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class VideoVisualizer:
    """Class for real-time video visualization"""
    
    def __init__(self, detection_visualizer: DetectionVisualizer):
        """Initialize video visualizer"""
        self.detection_viz = detection_visualizer
        self.frame_count = 0
        self.fps_counter = []
        
    def process_frame(self, 
                     frame: np.ndarray,
                     detections: Dict,
                     show_fps: bool = True,
                     show_stats: bool = True) -> np.ndarray:
        """
        Process single video frame with detections
        
        Args:
            frame: Input frame
            detections: Detection results
            show_fps: Whether to display FPS
            show_stats: Whether to display detection statistics
            
        Returns:
            Processed frame with visualizations
        """
        # Draw detections
        result_frame = self.detection_viz.draw_detections(frame, detections)
        
        # Add FPS counter
        if show_fps:
            result_frame = self._add_fps_counter(result_frame)
        
        # Add detection statistics
        if show_stats:
            result_frame = self._add_detection_stats(result_frame, detections)
        
        self.frame_count += 1
        return result_frame
    
    def _add_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """Add FPS counter to frame"""
        import time
        current_time = time.time()
        
        if len(self.fps_counter) >= 30:  # Keep last 30 measurements
            self.fps_counter.pop(0)
        
        if hasattr(self, '_last_time'):
            fps = 1.0 / (current_time - self._last_time)
            self.fps_counter.append(fps)
            avg_fps = np.mean(self.fps_counter)
            
            cv2.putText(frame, f'FPS: {avg_fps:.1f}',
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 0), 2)
        
        self._last_time = current_time
        return frame
    
    def _add_detection_stats(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Add detection statistics to frame"""
        height, width = frame.shape[:2]
        
        # Count detections above threshold
        scores = detections.get('scores', [])
        high_conf_count = sum(1 for score in scores 
                            if score >= self.detection_viz.confidence_threshold)
        
        stats_text = [
            f'Frame: {self.frame_count}',
            f'Detections: {high_conf_count}',
            f'Total Objects: {len(scores)}'
        ]
        
        # Draw stats in top-right corner
        for i, text in enumerate(stats_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text,
                       (width - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        return frame

def save_detection_results(detections: Dict, 
                         image_path: str,
                         output_path: str,
                         format: str = 'json'):
    """
    Save detection results to file
    
    Args:
        detections: Detection results dictionary
        image_path: Path to original image
        output_path: Path to save results
        format: Output format ('json', 'txt', 'xml')
    """
    import json
    import xml.etree.ElementTree as ET
    from pathlib import Path
    
    if format == 'json':
        result = {
            'image_path': image_path,
            'detections': {
                'boxes': detections.get('boxes', []).tolist() if hasattr(detections.get('boxes', []), 'tolist') else detections.get('boxes', []),
                'scores': detections.get('scores', []).tolist() if hasattr(detections.get('scores', []), 'tolist') else detections.get('scores', []),
                'classes': detections.get('classes', []).tolist() if hasattr(detections.get('classes', []), 'tolist') else detections.get('classes', [])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    elif format == 'txt':
        # YOLO format: class_id center_x center_y width height confidence
        with open(output_path, 'w') as f:
            boxes = detections.get('boxes', [])
            scores = detections.get('scores', [])
            classes = detections.get('classes', [])
            
            for box, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                f.write(f"{int(class_id)} {center_x} {center_y} {width} {height} {score}\n")

# Example usage and testing functions
if __name__ == "__main__":
    # Test the visualization utilities
    
    # Sample data for testing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_detections = {
        'boxes': np.array([[100, 100, 200, 200], [300, 150, 450, 300]]),
        'scores': np.array([0.85, 0.92]),
        'classes': np.array([0, 1])
    }
    
    class_names = ['person', 'car', 'bicycle', 'dog']
    
    # Initialize visualizers
    det_viz = DetectionVisualizer(class_names)
    metrics_viz = MetricsVisualizer()
    video_viz = VideoVisualizer(det_viz)
    
    print("Visualization utilities created successfully!")
    print("Classes supported:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Test detection visualization
    result_image = det_viz.draw_detections(test_image, test_detections)
    print(f"Detection visualization test: {result_image.shape}")
    
    # Test metrics visualization
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.25]
    val_losses = [0.9, 0.65, 0.45, 0.35, 0.3]
    fig = metrics_viz.plot_training_curves(train_losses, val_losses)
    print("Training curves visualization test completed")
    
    print("\nVisualization utilities are ready to use!")
