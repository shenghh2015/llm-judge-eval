import os
import glob


def new_dir(folder_path):
  """
    Construct a new folder
    """
  if os.path.exists(folder_path):
    print(f"{folder_path} has already existed!")
  else:
    os.makedirs(folder_path)
    print(f"{folder_path} created!")


def find_latest(path):
  """
    Find the latest file/folder in the path with the format xx.xx.date
    """
  searched = sorted(glob.glob(path))
  latest = searched[-1] if len(searched) != 0 else ''
  return latest
