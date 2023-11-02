
from PIL import Image

import os
import os.path as osp
import glob 
import numpy as np
import pandas as pd

from pathlib import Path
from tkinter import filedialog
from sklearn.cluster import KMeans
from tqdm import tqdm 
from shutil import copyfile

import matplotlib.pyplot as plt
import seaborn as sns 
import tkinter as tk
import io
import threading

import time

sns.set(style="darkgrid")

start_time = time.time()

def get_center_crop(img, percent=50):
    width, height = img.size
    new_width = width * percent // 100
    new_height = height * percent // 100
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return img.crop((left, top, right, bottom))

def get_grid_threshold(img):
    cropped_img = img
    img_array = np.array(cropped_img)
    gradient_x = np.abs(np.gradient(img_array, axis=1))
    gradient_y = np.abs(np.gradient(img_array, axis=0))
    avg_gradient = (np.mean(gradient_x) + np.mean(gradient_y)) / 2
    return avg_gradient

def get_png_size(img):
    byte_io = io.BytesIO()
    img.save(byte_io, format='PNG')
    return len(byte_io.getvalue())

def compute_gradient(img_files, gradients, event):
    for i, img_file in enumerate(img_files):
        img = Image.open(img_file)     
        avg_gradient = get_grid_threshold(img)
        gradients[i] = avg_gradient
        img.close()
    event.set()

def compute_png_size(img_files, png_sizes, event):
    for i, img_file in enumerate(img_files):
        img = Image.open(img_file)     
        png_size = get_png_size(img)
        png_sizes[i] = png_size
        img.close()
    event.set()

root = tk.Tk()
root.withdraw()
input_dir = filedialog.askdirectory(title='Select Input Directory')
if not input_dir:
    raise Exception("No input directory selected!")

output_dir = osp.join(input_dir, 'output_data')
output_dir_tmp = osp.join(output_dir, 'tmp')
output_dir_bmps = osp.join(output_dir, 'bmps')
output_dir_jpgs = osp.join(output_dir, 'jpgs')

for _out in [output_dir_tmp, output_dir_bmps, output_dir_jpgs]:
    if not osp.exists(_out):
        os.makedirs(_out)

folders = [folder for folder in glob.glob(osp.join(input_dir, "**")) if osp.isdir(folder) and 'output_data' not in folder]
features = []

for folder in folders:
    img_files = sorted(glob.glob(osp.join(folder, '*.bmp')))
    gradients = [None] * len(img_files)
    png_sizes = [None] * len(img_files)
    gradient_event = threading.Event()
    png_size_event = threading.Event()
    gradient_thread = threading.Thread(target=compute_gradient, args=(img_files, gradients, gradient_event))
    png_size_thread = threading.Thread(target=compute_png_size, args=(img_files, png_sizes, png_size_event))
    gradient_thread.start()
    png_size_thread.start()
    gradient_event.wait()
    png_size_event.wait()
    features.extend(list(zip(gradients, png_sizes)))

features = np.array(features)
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(features)

# 분류된 라벨 별로 scatter 그리기
for i in range(2):
    plt.scatter(features[kmeans.labels_ == i][:, 0], features[kmeans.labels_ == i][:, 1], 
                label=f'Cluster {i+1}', marker='os^'[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='X', label='Centers')
plt.xlabel('Gradient Value')
plt.ylabel('PNG Size')
plt.title('2D Clustering based on Gradient and PNG Size')
plt.legend()
plt.show()

# Label을 기준으로 Real bmp와 Fake bmp를 분류
# 예를 들어, label 0을 real bmp로, 라벨 1을 fake bmp로 가정합니다.
REAL_BMP_LABEL = 0
FAKE_BMP_LABEL = 1

df_dict = {'filename': [], 'avg gradient': [], 'format': []}
cnt_bmp, cnt_jpg = 0, 0
cnt = 0

for folder_idx, folder in enumerate(folders):
    img_files = sorted(glob.glob(osp.join(folder, '*.bmp')))
    for file_idx, img_file in tqdm(enumerate(img_files), desc=f"{folder}-{cnt_bmp}-{cnt_jpg}"):
        img = Image.open(img_file)     
        filename = osp.basename(osp.splitext(img_file)[0])
        df_dict['filename'].append(filename)
        avg_gradient = get_grid_threshold(img)
        df_dict['avg gradient'].append(avg_gradient)
        
        assigned_label = kmeans.labels_[cnt]
        if assigned_label == REAL_BMP_LABEL:
            destination_bmp = osp.join(output_dir_bmps, folder.replace(input_dir, '').strip(os.sep), filename + '.bmp')
            os.makedirs(osp.dirname(destination_bmp), exist_ok=True)
            if not osp.exists(destination_bmp) and img_file != destination_bmp:
                copyfile(img_file, destination_bmp)
            cnt_bmp += 1
            df_dict['format'].append('bmp')
        elif assigned_label == FAKE_BMP_LABEL:
            destination_jpg = osp.join(output_dir_jpgs, folder.replace(input_dir, '').strip(os.sep), filename + '.bmp')
            os.makedirs(osp.dirname(destination_jpg), exist_ok=True)
            if not osp.exists(destination_jpg) and img_file != destination_jpg:
                copyfile(img_file, destination_jpg)
            cnt_jpg += 1
            df_dict['format'].append('jpg')
        
        img.close()
        cnt += 1
        if cnt % 50 == 0:
            df = pd.DataFrame(df_dict)        
            df.to_csv(osp.join(output_dir, 'analyze.csv'), index=False)
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot the average gradient per image
            axes[0].plot(df['avg gradient'])
            axes[0].set_xlabel('Number of Images')
            axes[0].set_ylabel('Average Gradient')
            axes[0].set_title("Average Gradient per Image")
            
            # Boxplot for BMP vs JPG based on the average gradient
            sns.boxplot(x='format', y='avg gradient', data=df, ax=axes[1])
            
            plt.suptitle(f"Image Analysis of {input_dir}")
            plt.savefig(osp.join(output_dir, 'analyze.jpg'))
            plt.close()

end_time = time.time()
print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
