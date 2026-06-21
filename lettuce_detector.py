import cv2
import numpy as np
import onnxruntime as ort

class LettuceDetector:
    def __init__(self, model_path, conf_threshold=0.5, crop_w=1000, crop_h=1000):
        """
        Initializes the detector. Do this ONLY ONCE when your main.py starts up.
        """
        print(f"[Detector] Loading ONNX model from {model_path}...")
        # CPUExecutionProvider is highly optimized for Raspberry Pi ARM processors
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        self.conf_threshold = conf_threshold
        self.crop_w = crop_w
        self.crop_h = crop_h
        
        # Dictionary mapping your category IDs to names
        self.class_names = {
            1: "Anthracnose", 2: "Aphid", 3: "Bacterial Soft Rot",
            4: "Big Vein", 5: "Downy Mildew", 6: "Healthy",
            7: "Leaf Miner", 8: "Lettuce Mosaic Virus", 9: "Other Disease",
            10: "Other Pests", 11: "Powdery Mildew", 12: "Septoria Leaf Spot",
            13: "Thrip", 14: "Tip Burn", 15: "Whitefly"
        }
        print("[Detector] Model loaded and ready.")

    def _preprocess(self, img):
        """Internal method to convert the cropped image into a PyTorch-friendly tensor."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (448, 448))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        input_tensor = np.expand_dims(img_transposed, axis=0)
        return input_tensor

    def detect(self, original_img, draw_results=False, is_yuyv=False):
        """
        Receives a cv2 image, scans the center, and returns the detection data.
        - Set draw_results=True if you want it to return an annotated image as well.
        - Set is_yuyv=True if your camera captures raw YUYV frames.
        """
        if original_img is None:
            print("[Detector] Error: Received an empty image.")
            return [], None

        # Convert YUYV to standard BGR if necessary before any processing
        if is_yuyv:
            try:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_YUV2BGR_YUYV)
            except Exception as e:
                print(f"[Detector] Warning: YUYV conversion failed. Error: {e}")

        orig_h, orig_w = original_img.shape[:2]
        
        # Calculate center crop coordinates
        cx, cy = orig_w // 2, orig_h // 2
        half_cw, half_ch = self.crop_w // 2, self.crop_h // 2
        
        # Ensure crop dimensions stay within the image
        start_x = max(0, cx - half_cw)
        start_y = max(0, cy - half_ch)
        end_x = min(orig_w, cx + half_cw)
        end_y = min(orig_h, cy + half_ch)
        
        actual_crop_w = end_x - start_x
        actual_crop_h = end_y - start_y
        
        # Extract the middle scanning zone
        cropped_img = original_img[start_y:end_y, start_x:end_x]
        
        # Preprocess and run ONNX inference
        input_tensor = self._preprocess(cropped_img)
        outputs = self.session.run(None, {'input_image': input_tensor})
        
        boxes, labels, scores = outputs[0], outputs[1], outputs[2]
        
        # Scale ratios for mapping back to the cropped area
        scale_x = actual_crop_w / 448.0
        scale_y = actual_crop_h / 448.0
        
        annotated_img = None
        if draw_results:
            annotated_img = original_img.copy()
            # Draw the Blue Scanning Zone box
            cv2.rectangle(annotated_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 4)
            cv2.putText(annotated_img, "Scanning Zone", (start_x + 10, start_y + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        detections_data = []

        for i in range(len(scores)):
            if scores[i] >= self.conf_threshold:
                # Map coordinates back to crop size
                x1_crop = int(boxes[i][0] * scale_x)
                y1_crop = int(boxes[i][1] * scale_y)
                x2_crop = int(boxes[i][2] * scale_x)
                y2_crop = int(boxes[i][3] * scale_y)
                
                # Offset to absolute 4K image position
                x1 = x1_crop + start_x
                y1 = y1_crop + start_y
                x2 = x2_crop + start_x
                y2 = y2_crop + start_y
                
                label_id = int(labels[i])
                score = float(scores[i])
                class_name = self.class_names.get(label_id, f"Unknown ({label_id})")
                
                # Append raw data to our return list
                detections_data.append({
                    "class": class_name,
                    "confidence": score,
                    "box": [x1, y1, x2, y2]
                })
                
                if draw_results:
                    # Draw the Green Bounding Box
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    # Draw the Label
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.rectangle(annotated_img, (x1, y1 - 40), (x1 + len(label_text) * 18, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_img, label_text, (x1 + 5, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return detections_data, annotated_img