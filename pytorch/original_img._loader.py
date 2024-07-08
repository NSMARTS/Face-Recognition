import numpy as np
import os
import shutil
from tqdm import tqdm

dir_path = '../kface'

print(dir_path)

# print(os.listdir(dir_path))

subfolders = sorted(
        [file.path for file in os.scandir(dir_path) if file.is_dir()])

# print(subfolders)

all_img = []
for idx, folder in tqdm(enumerate(subfolders)):
        # print(idx)
        # print(folder)
        tmp = 0
        for file in sorted(os.listdir(folder)):
                all_img.append(folder+"/"+file)



print(len(all_img))
print(all_img[0])