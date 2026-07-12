import os
import sys
import cv2
import time
import easyocr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def benchmark():
    img_path = "dataset/img/001.jpg"
    if not os.path.exists(img_path):
        # find any image in dataset/img
        img_dir = "dataset/img"
        files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        if not files:
            print("No images found to benchmark.")
            return
        img_path = os.path.join(img_dir, files[0])

    print(f"Benchmarking image: {img_path}")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    print("Initializing Reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Warm up
    print("Warming up EasyOCR...")
    _ = reader.readtext(img_rgb, canvas_size=640)
    
    sizes = [640, 800, 1000, 1200, 1600, 2560]
    for size in sizes:
        t0 = time.time()
        results = reader.readtext(img_rgb, canvas_size=size)
        elapsed = time.time() - t0
        
        num_boxes = len(results)
        sample_texts = [r[1] for r in results[:5]]
        print(f"\nCanvas Size: {size}")
        print(f"  Time taken: {elapsed:.3f} seconds")
        print(f"  Boxes found: {num_boxes}")
        print(f"  Sample texts: {sample_texts}")

if __name__ == "__main__":
    benchmark()
