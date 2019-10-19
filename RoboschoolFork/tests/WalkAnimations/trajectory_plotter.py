import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import math

def interpolate(original, steps):
    final = [None] * steps
    n = (len(original)-1)/(len(final)-1)
    for i in range(len(final)):
        low_index = str(int(math.floor(n * i)))
        high_index = str( int(math.ceil(n * i)) )
        low = np.array( original[ low_index ] )
        high = np.array( original[ high_index ] )
        delta_pos = high - low
        delta_t = n * i - math.floor(n * i)
        final[i] = low + delta_pos * delta_t
    final = np.vstack(final)
    return final

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
xs=[]
ys=[]
zs=[]
t=[]
part_names = ["mixamorig_RightForeArm",
"mixamorig_LeftForeArm",
"mixamorig_Spine2",
"mixamorig_RightThigh",
"mixamorig_LeftThigh",
"mixamorig_RightTibia",
"mixamorig_LeftTibia",
"mixamorig_RightHand",
"mixamorig_LeftHand",
"mixamorig_RightUpLeg",
"mixamorig_LeftUpLeg",
"mixamorig_RightToeBase",
"mixamorig_LeftToeBase", ]
file_names = [
    "Walk1.1.json",
    "Walk1.2.json",
    "Walk2.1.json",
    "Walk2.2.json",
    "Walk3.1.json",
    "Walk3.2.json",
]
InterpolatedWalkAnimations = []
fig = {}
for filename in file_names:
    fig[filename] = plt.figure()
    ax = fig[filename].add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-60, 60)
    ax.set_ylim3d(-60,60)
    ax.set_zlim3d(10,-120)

    if(int(filename.split(".")[-2]) == 2):
        mirrorx = True
    else:
        mirrorx = False
    mirrorx = False
    with open(filename,"r") as file:
        paths = json.load(file)
    paths["mixamorig_RightThigh"] = {}
    paths["mixamorig_LeftThigh"] = {}
    paths["mixamorig_RightTibia"] = {}
    paths["mixamorig_LeftTibia"] = {}

    for i in paths["mixamorig_RightToe_End"].keys():
        paths["mixamorig_RightThigh"][i] = list((np.array(paths["mixamorig_RightLeg"][i]) +
            np.array(paths["mixamorig_RightUpLeg"][i]))/2)
        paths["mixamorig_LeftThigh"][i] = list((np.array(paths["mixamorig_LeftLeg"][i]) +
            np.array(paths["mixamorig_LeftUpLeg"][i]))/2)
        paths["mixamorig_RightTibia"][i] = list((np.array(paths["mixamorig_RightLeg"][i]) +
            np.array(paths["mixamorig_RightFoot"][i]))/2)
        paths["mixamorig_LeftTibia"][i] = list((np.array(paths["mixamorig_LeftLeg"][i]) +
            np.array(paths["mixamorig_LeftFoot"][i]))/2)

    important_paths = {}
    center = interpolate(paths["mixamorig_Hips"], 15)
    for part_name in part_names:
        a = interpolate(paths[part_name], 15)
        if mirrorx:
            a[:,0] = a[:,0] * -1
        a = a - center

        a[:, 1], a[:, 2] = a[:, 2], a[:, 1].copy()#swap y and z because axis are different in roboschool
        a[:, 0], a[:, 1] = a[:, 1], a[:, 0].copy()#swap x and y because axis are different in roboschool
        a[:, 1] = a[:,1] * 3  #make everything a bit wider
        important_paths[part_name] = list(appendSpherical_np(a).tolist())
    InterpolatedWalkAnimations.append(important_paths)


    new_important_paths = {}
    for part_name in part_names:
        new_important_paths[part_name] = np.array(important_paths[part_name])
    for part_name in part_names:
        xs = new_important_paths[part_name][:,0]
        ys = new_important_paths[part_name][:,1]
        zs= new_important_paths[part_name][:,2]
        # x= new_important_paths["mixamorig_Spine2"][:,0]
        # y= new_important_paths["mixamorig_Spine2"][:,1]
        # z= new_important_paths["mixamorig_Spine2"][:,2]
        ax.quiver(0, 0, 0, xs, ys, zs)
        ax.scatter(xs, ys, zs)
    plt.pause(0.01)
print(InterpolatedWalkAnimations)
with open("AnimationsProcessed.json", "w") as file:
    file.write(json.dumps(InterpolatedWalkAnimations, indent=2, sort_keys=True))
plt.show()
