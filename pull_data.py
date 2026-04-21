import kagglehub
import shutil
from pathlib import Path
    
repo_root = Path(__file__).resolve().parent
dataset_root = Path(kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces"))
print("Downloaded to:", dataset_root)

source = dataset_root / "real_vs_fake" / "real-vs-fake"
dest = repo_root / "data" / "real_vs_fake"

if not dest.exists():
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    print("Copied to:", dest)
else:
    print("Already exists at:", dest)