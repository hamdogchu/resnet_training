import os
import cv2
from lettuce_detector import LettuceDetector
# from supabase_uploader import CloudUploader

def run_batch_inference(input_dir, output_dir, model_path="lettuce_student_resnet18.onnx", is_yuyv=False):
    os.makedirs(output_dir, exist_ok=True)
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.raw', '.yuv')
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if f.lower().endswith(valid_extensions) or '.' not in f]

    if not image_paths:
        print(f"[Batch Processor] No images found in '{input_dir}'. Skipping inference.")
        return {} # Return an empty dictionary if no images

    print(f"\n[Batch Processor] Waking up AI. Processing {len(image_paths)} images...")
    detector = LettuceDetector(model_path=model_path, conf_threshold=0.5, crop_w=1000, crop_h=1000)

    # Dictionary to hold the detection data for the database
    wave_results = {}

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f" -> Analyzing {filename}...")

        img = cv2.imread(img_path)
        if img is None:
            continue

        apply_yuyv = is_yuyv
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            apply_yuyv = False

        detections, annotated_img = detector.detect(img, draw_results=True, is_yuyv=apply_yuyv)

        if annotated_img is not None:
            save_name = filename if '.' in filename else f"{filename}.jpg"
            final_filename = f"scanned_{save_name}"
            out_path = os.path.join(output_dir, final_filename)
            cv2.imwrite(out_path, annotated_img)
            
            # Map the final saved filename to its AI detections
            wave_results[final_filename] = detections

        if detections:
            print(f"    Found {len(detections)} target(s):")
            for d in detections:
                print(f"      - {d['class']} ({d['confidence']*100:.1f}%)")
        else:
            print("    No issues detected (Healthy).")

    print(f"[Batch Processor] Wave inference complete.\n")
    
    # Return the data payload so the uploader can read it
    return wave_results

if __name__ == "__main__":
    ai_results = run_batch_inference(
                    input_dir="sample_captured", 
                    output_dir="processed_captures",
                    model_path="lettuce_student_resnet18.onnx",
                    # Set to True if capture_image() outputs raw YUYV instead of standard JPG/PNG formats
                    is_yuyv=False 
                )
    print("\n[PHASE 3] Syncing to Supabase Cloud...")
    '''
    uploader = CloudUploader(
        supabase_url='https://eiqtrqnioobwslzntkdo.supabase.co', 
        supabase_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVpcXRycW5pb29id3Nsem50a2RvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODA4OTk1NDMsImV4cCI6MjA5NjQ3NTU0M30.cUNW3nWp3D6iO2IjLsq1-8zLNz4_3i1bgEe8dy6Dyag', 
        bucket_name='scans'
    )
    uploader.upload_wave("processed_captures", ai_results)
    '''
    
    
