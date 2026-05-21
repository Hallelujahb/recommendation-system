from surprise import Dataset
from surprise.builtin_datasets import BUILTIN_DATASETS
import os, shutil

# This downloads ml-100k directly from GroupLens via surprise's own mirror
data = Dataset.load_builtin('ml-100k')

# Find where surprise saved it
surprise_data_path = os.path.join(os.path.expanduser('~'), '.surprise_data', 'ml-100k', 'ml-100k')
print("Surprise saved files to:", surprise_data_path)
print("Files available:", os.listdir(surprise_data_path))

# Copy to your project folder
dest = r"C:\Users\halle\Documents\iCog\Trainings\recommendation\ml-100k"
shutil.copytree(surprise_data_path, dest, dirs_exist_ok=True)
print("Copied to:", dest)
print("Files in project:", os.listdir(dest))