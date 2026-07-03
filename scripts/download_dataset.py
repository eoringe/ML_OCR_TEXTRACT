import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import stat
import time

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree to remove read-only attributes on Windows files.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        # Ignore if we still can't remove it, shutil.rmtree will raise it
        pass

def clean_path(path):
    if os.path.exists(path):
        flush_print(f"Cleaning up: {path}")
        if os.path.isdir(path):
            shutil.rmtree(path, onerror=remove_readonly)
            # Second attempt if still exists
            if os.path.exists(path):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                except Exception:
                    pass
        else:
            try:
                os.remove(path)
            except Exception:
                pass

def download_file_with_retries(url, path, max_retries=5):
    """
    Downloads a file with retries for network resilience.
    """
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            flush_print(f"Attempt {attempt}/{max_retries} to download dataset ZIP...")
            if os.path.exists(path):
                os.remove(path)
                
            with urllib.request.urlopen(req, timeout=30) as response:
                meta = response.info()
                content_length = meta.get("Content-Length")
                total_size = int(content_length) if content_length else None
                
                if total_size:
                    flush_print(f"  Total size: {total_size / (1024*1024):.2f} MB")
                else:
                    flush_print("  Total size: Unknown")
                    
                bytes_so_far = 0
                block_size = 1024 * 1024  # 1MB blocks
                
                with open(path, 'wb') as out_file:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        bytes_so_far += len(buffer)
                        out_file.write(buffer)
                        if total_size:
                            percent = (bytes_so_far / total_size) * 100
                            flush_print(f"  Downloaded {bytes_so_far / (1024*1024):.2f} MB ({percent:.1f}%)")
                        else:
                            flush_print(f"  Downloaded {bytes_so_far / (1024*1024):.2f} MB")
            
            flush_print("ZIP download successfully completed!")
            return True
            
        except Exception as e:
            flush_print(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                sleep_time = attempt * 5
                flush_print(f"  Waiting {sleep_time} seconds before retrying...")
                time.sleep(sleep_time)
            else:
                flush_print("  All download attempts failed.")
                
    return False

def download_sroie():
    # Make sure we run from the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    flush_print("==================================================")
    flush_print("         SROIE Receipt Dataset Downloader        ")
    flush_print("==================================================")
    
    temp_dir = "temp_sroie"
    dataset_dir = "dataset"
    
    # Pre-clean everything to prevent locks
    for path in [temp_dir, "temp_zip_extract", "sroie_master.zip"]:
        clean_path(path)
        
    download_success = False
    zip_path = "sroie_master.zip"
    extract_dir = "temp_zip_extract"
    zip_url = "https://github.com/zzzDavid/ICDAR-2019-SROIE/archive/refs/heads/master.zip"
    
    # Method 1: Download ZIP
    download_success = download_file_with_retries(zip_url, zip_path, max_retries=5)
    
    if download_success:
        try:
            flush_print("Extracting ZIP archive...")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
                
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            nested_folder = os.path.join(extract_dir, "ICDAR-2019-SROIE-master")
            if os.path.exists(os.path.join(nested_folder, "data")):
                os.makedirs(temp_dir, exist_ok=True)
                shutil.move(os.path.join(nested_folder, "data"), os.path.join(temp_dir, "data"))
                flush_print("ZIP extraction and directory alignment successful.")
            else:
                flush_print("Error: Could not find 'data' folder inside the extracted ZIP.")
                download_success = False
        except Exception as e:
            flush_print(f"Extraction failed: {e}")
            download_success = False
        finally:
            # Clean up ZIP files and folders
            clean_path(zip_path)
            clean_path(extract_dir)
            
    # Method 2: Git Clone (Fallback)
    if not download_success:
        flush_print("\n[Method 2 Fallback] Cloning repository via Git...")
        clean_path(temp_dir) # Ensure temp_dir is fully cleared
        try:
            repo_url = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"
            flush_print(f"Cloning {repo_url} (depth=1)...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, temp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and os.path.exists(os.path.join(temp_dir, "data")):
                download_success = True
                flush_print("Repository cloned successfully!")
            else:
                flush_print(f"Git clone failed with code {result.returncode}.")
                flush_print(f"Error output: {result.stderr}")
        except Exception as e:
            flush_print(f"Git clone encountered an exception: {e}")
            
    # If successful, move the data directory to the final destination
    if download_success:
        try:
            temp_data_dir = os.path.join(temp_dir, "data")
            if os.path.exists(dataset_dir):
                clean_path(dataset_dir)
                
            flush_print(f"Moving dataset files into final path '{dataset_dir}/'...")
            shutil.move(temp_data_dir, dataset_dir)
            flush_print("\nDataset successfully loaded and structured!")
            flush_print(f"Path: {os.path.abspath(dataset_dir)}")
            
            # Print directory breakdown
            subdirs = ['img', 'box', 'key']
            for sd in subdirs:
                sd_path = os.path.join(dataset_dir, sd)
                if os.path.exists(sd_path):
                    count = len([f for f in os.listdir(sd_path) if os.path.isfile(os.path.join(sd_path, f))])
                    flush_print(f"  - dataset/{sd}/: {count} files found")
            flush_print("==================================================")
            
        except Exception as e:
            flush_print(f"Error structuring dataset folder: {e}")
        finally:
            # Clean up temp folder
            clean_path(temp_dir)
    else:
        flush_print("\n[CRITICAL ERROR] Failed to acquire the dataset via all methods.")
        sys.exit(1)

if __name__ == "__main__":
    download_sroie()
