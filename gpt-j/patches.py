from pathlib import Path
import os
import re
import shutil
from distutils.sysconfig import get_python_lib

def patch_bits_and_bytes(
    desired_cuda_version: int
):
    site_packages = get_python_lib()
    folder_path = site_packages + "/bitsandbytes/"
    files = os.listdir(folder_path)

    pattern = r'libbitsandbytes_cuda(\d+)'

    for file in files:
        if file.startswith('libbitsandbytes_cuda') and file.endswith('.so'):
            match = re.search(pattern, file)
            if match:
                version_str = match.group(1)
                version = int(version_str)  
                if version > desired_cuda_version:
                    file_path = os.path.join(folder_path, file)
                    os.remove(file_path)  
                elif version == desired_cuda_version:
                    file_path = os.path.join(folder_path, file)
                    new_file_path = os.path.join(folder_path, 'libbitsandbytes_cpu.so')
                    shutil.copyfile(file_path, new_file_path)