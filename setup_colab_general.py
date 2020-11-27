import requests
import sys
from tqdm.auto import tqdm
import numpy as np
import os

def download_github_content(path, filename, chnksz=1000):
    """
    Functions that downloads content from python_utils github
    repository
    :param path: the file path
    :param filename: the filename
    :param chnksz: the chunk size
    """
    url = f"https://raw.githubusercontent.com/jpcano1/python_utils/main/{path}"
    
    try:
        r = requests.get(url, stream=True)
    except Exception as e:
        print(f"Error de conexión con el servidor: {e}")
        sys.exit()
        
    with open(filename, "wb") as f:
        try:
            total = int(np.ceil(int(r.headers.get("content-length"))/chnksz))
        except ArithmeticError:
            total = 0

        gen = r.iter_content(chunk_size=chnksz)

        for pkg in tqdm(gen, total=total, unit="KB"):
            f.write(pkg)

        f.close()
        r.close()
    return

def setup_general():
    """
    Function that enables the general functions in google colab
    """
    os.makedirs("utils", exist_ok=True)
    with open("utils/__init__.py", "wb") as f:
        f.close()

    download_github_content("utils/general.py", "utils/general.py")
    print("General Functions Enabled Successfully")