import os

# Get the root directory of the repository
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct the path to the raw directory
raw_dir = os.path.join(repo_root, 'data', 'raw')

# List the files in the raw directory
filenames = os.listdir(raw_dir)
