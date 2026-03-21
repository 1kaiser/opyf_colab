#!/usr/bin/env python3
import sys, os
import opyf
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.interpolate import interp1d
import rasterio as rio
import csv
from tqdm import tqdm

# Constants
BASE_DIR = "opyf_colab/tests/Test_Brague_flood"
os.chdir(BASE_DIR)

print("Starting Full LSPIV Analysis for Brague Flood...")

# --- 1139 Analysis ---
print("\n--- Processing IMG_1139.MOV ---")
os.chdir("1139")
# The video is in the parent directory
video_file_1139 = "../IMG_1139.MOV"
video = opyf.videoAnalyzer(video_file_1139)

video.set_vecTime(Ntot=50, starting_frame=200)
video.set_interpolationParams(Sharpness=2)
video.set_goodFeaturesToTrackParams(qualityLevel=0.01)

mask = cv2.imread('mask_1139.png')
if mask is not None:
    A = mask > 100
    video.set_stabilization(mask=A[:,:,0], mute=False)
else:
    print("Warning: mask_1139.png not found")

image_points = np.array([
    (355,429), (1338,350), (99, 562), (1673, 364)
], dtype="double")
 
model_points = np.array([
    (30.13,-8.28,0), (32.88,-28.08,0), (20.46, -4.47, 0.4), (21.32, -27.14, 0.4)
], dtype="double")

abs_or = model_points[0]
model_points = model_points - model_points[0]

video.set_birdEyeViewProcessing(image_points, model_points, [-12, 4, -32.], 
                                 rotation=np.array([[1., 0, 0],[0,-1,0],[0,0,-1.]]), 
                                 scale=True, framesPerSecond=30)

video.set_vlim([0, 10])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15, range_Vx=[0.01,10])
video.filterAndInterpolate()

X1139 = np.copy(video.Xaccu + abs_or[0:2])
V1139 = np.copy(video.Vaccu)
norm1139 = (V1139[:, 0]** 2 + V1139[:, 1]** 2)** 0.5

# --- 1142 Analysis ---
print("\n--- Processing IMG_1142.MOV ---")
os.chdir("../1142")
video_file_1142 = "../IMG_1142.MOV"
video = opyf.videoAnalyzer(video_file_1142)

video.set_vecTime(Ntot=50, starting_frame=0)
video.set_goodFeaturesToTrackParams(qualityLevel=0.001)
video.set_interpolationParams(Sharpness=2)

mask = cv2.imread('mask_1142.png')
if mask is not None:
    A = mask > 100
    video.set_stabilization(mask=A[:,:,0], mute=False)
else:
    print("Warning: mask_1142.png not found")

image_points = np.array([
    (830,564), (1480, 594), (1750,800), (0,616), (369,565)
], dtype="double")
 
model_points = np.array([
    (-2,-10.5,0), (0, 0., 0), (21.2, -5, 0), (7.4,-21.7,0), (-2.6,-19.9,0)
], dtype="double")

video.set_birdEyeViewProcessing(image_points, model_points, [12, 7, -30.], 
                                 rotation=np.array([[1., 0, 0],[0,-1,0],[0,0,-1.]]), 
                                 scale=True, framesPerSecond=30)
video.set_vlim([0,10])
video.extractGoodFeaturesDisplacementsAccumulateAndInterpolate()
video.set_filtersParams(maxDevInRadius=1.5, RadiusF=0.15, range_Vx=[0.01,10])
video.filterAndInterpolate()

X1142 = np.copy(video.Xaccu)
V1142 = np.copy(video.Vaccu)
norm1142 = (V1142[:,0]**2 + V1142[:,1]**2)**0.5

# --- Combine and Export ---
print("\n--- Combining Data ---")
Xtot = np.append(X1142, X1139, axis=0)
Vtot = np.append(V1142, V1139, axis=0)
Ntot = np.append(norm1142, norm1139, axis=0)

export_H5 = '../export_1142_1139_high_res.h5'
opyf.hdf5_WriteUnstructured2DTimeserie(export_H5, [0], [Xtot], [Vtot])

# --- Bathymetry & Discharge Calculation ---
print("\n--- Loading Bathymetry (MNT.xyz) ---")
os.chdir("..")
datas = []
with open('MNT.xyz', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    index = 0
    nextIt = int(np.random.uniform() * 20)
    for row in reader:
        if index == nextIt:
            temp = [float(item) for item in row]
            datas.append(temp)
            nextIt += 1 + int(np.random.uniform() * 99)
        index += 1
datas = np.array(datas)

x0, y0 = 1030760.6875, 6289057
terrain = np.copy(datas)
terrain[:, 0] -= x0
terrain[:, 1] -= y0

X1, X2 = np.array([16., -2]), np.array([11., -23.])    
Npoints = 1000
dr = LA.norm(X2 - X1) / Npoints
dvecr = (X2 - X1) / LA.norm(X1 - X2) * dr
vecX = np.array([X1 + dvecr * ir for ir in range(Npoints)])
parm_interp = {'kernel': 'Gaussian', 'Radius': 5*dr, 'Sharpness': 2.}
V1 = opyf.Interpolate.npInterpolateVTK2D(terrain[:, 0:2], terrain[:, 1:3], vecX, ParametreInterpolatorVTK=parm_interp)

bathy_new = np.zeros((len(V1), 3))
bathy_new[:, 0:2] = vecX[:, 0:2]
bathy_new[:, 2] = V1[:, 1]

fz = interp1d(bathy_new[:, 1], bathy_new[:, 2], fill_value="extrapolate")
XY_ext = np.array([X1 + dvecr * (ir - 100) for ir in range(Npoints + 200)])
bathy_ext = np.zeros((len(XY_ext), 3))
bathy_ext[:, 0:2] = XY_ext[:, 0:2]
bathy_ext[:, 2] = fz(XY_ext[:, 1])

zwater = 14.4
bathy_ext[:, 2] -= zwater

# --- Final Visualization and Discharge ---
print("\n--- Calculating Discharge & Plotting ---")
plt.figure(figsize=(10, 6))
cmapm = opyf.make_cmap_customized()
plt.scatter(Xtot[:, 0], Xtot[:, 1], c=Ntot, alpha=0.6, s=1, cmap=cmapm)
plt.colorbar(label='Velocity (m/s)')
plt.axis('equal')

ParametreInterpolatorVTK = {'kernel': 'Gaussian', 'Radius': 2, 'Sharpness': 2.}
V = opyf.Interpolate.npInterpolateVTK2D(Xtot, Vtot, bathy_ext[:, 0:2], ParametreInterpolatorVTK)
bathy_z = bathy_ext[:, 2]
ind_im = np.where(bathy_z <= 0.)[0]

lengthTrans = LA.norm(bathy_ext[ind_im[0], 0:2] - bathy_ext[ind_im[-1], 0:2])
# Q = Mean(Velocity * Area)
v_norm = (V[ind_im, 0]**2 + V[ind_im, 1]**2)**0.5
Q = -np.mean(v_norm * bathy_z[ind_im]) * lengthTrans

plt.title(f'Brague Flood High-Precision Analysis - Q = {Q:.2f} m3/s')
plt.savefig('high_precision_analysis_report.png')
print(f"Analysis Complete. Calculated Discharge: {Q:.2f} m3/s")
print("Report saved to high_precision_analysis_report.png")

