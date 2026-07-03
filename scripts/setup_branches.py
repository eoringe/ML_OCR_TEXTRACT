import sys
import subprocess
import os

ROLES = [
    {"task": 1, "title": "Data Preprocessing & Loader", "branch_base": "task-1-data-pipeline", "file": "src/data_pipeline/prep.py"},
    {"task": 2, "title": "Text Detection Model", "branch_base": "task-2-text-detection", "file": "src/detection/detect.py"},
    {"task": 3, "title": "Text Recognition (OCR)", "branch_base": "task-3-text-recognition", "file": "src/recognition/recognize.py"},
    {"task": 4, "title": "Key Information Extraction & Post-Processing", "branch_base": "task-4-kie-parsing", "file": "src/kie/extract.py, src/post_processing/parser.py"},
    {"task": 5, "title": "API / Backend Service", "branch_base": "task-5-backend-api", "file": "src/backend/app.py"},
    {"task": 6, "title": "Frontend UI Dashboard", "branch_base": "task-6-frontend-ui", "file": "src/frontend/"},
    {"task": 7, "title": "Evaluation & MLOps CI/CD", "branch_base": "task-7-mlops-eval", "file": "src/utils/evaluate.py, Dockerfile"}
]

def run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def setup_branches(member_names):
    # Ensure correct count
    if len(member_names) != 7:
        print(f"Error: You must provide exactly 7 member names (received {len(member_names)}).")
        return
        
    print("==================================================")
    print("       Setting Up Team Branches for SROIE         ")
    print("==================================================")
    
    # 1. Ensure we are in project root and check git status
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 2. Check if we have uncommitted work
    success, out, err = run_cmd(["git", "status", "--porcelain"])
    # We want to check for untracked files or modifications except dataset directory
    lines = [line for line in out.strip().split("\n") if line and not line.strip().endswith("dataset/")]
    if lines:
        print("Warning: You have uncommitted changes in your repository (excluding dataset folder).")
        print("Please commit your initial skeleton files before setting up branches.")
        print("To commit: git add . && git commit -m \"feat: initial repository setup\"")
        # return

    # 3. Create branches
    for i, name in enumerate(member_names):
        role = ROLES[i]
        # Sanitize name to prevent bad git references
        sanitized_name = "".join([c if c.isalnum() else "-" for c in name.lower().strip()])
        branch_name = f"{role['branch_base']}-{sanitized_name}"
        
        print(f"\nMember {i+1}: {name}")
        print(f"  - Role:   {role['title']}")
        print(f"  - Branch: {branch_name}")
        print(f"  - File:   {role['file']}")
        
        # Git branch creation command
        success, out, err = run_cmd(["git", "branch", branch_name, "main"])
        if success:
            print(f"  -> Branch '{branch_name}' created successfully.")
        else:
            if "already exists" in err:
                print(f"  -> Branch '{branch_name}' already exists.")
            else:
                print(f"  -> Failed to create branch: {err.strip()}")
                
    print("\n==================================================")
    print("All branches set up! To check branches run: git branch")
    print("==================================================")

if __name__ == "__main__":
    if len(sys.argv) < 8:
        print("Usage: python scripts/setup_branches.py <name1> <name2> ... <name7>")
        print("Example: python scripts/setup_branches.py Alice Bob Charlie Dave Eve Frank Grace")
        sys.exit(1)
        
    names = sys.argv[1:8]
    setup_branches(names)
