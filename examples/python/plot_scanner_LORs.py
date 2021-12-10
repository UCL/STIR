# Demo of how to use STIR from python to plot line of responses of a blocksOnCylindrical scanner in different 2D 
# orientation and in 3Ds. The plots show the coordinate system used in STIR with labels L for left, R for right, 
# P for posterior, A for anterior, H for Head and F for feet. This is easily modifiable for Cylindrical scanner. 
# To run in "normal" Python, you would type the following in the command line
#  execfile('plot_scanner_LORs.py')
# In ipython, you can use
#  %run plot_scanner_LORs.py

# Copyright 2021 University College London
# Copyright 2021 National Phyisical Laboratory

# Author Daniel Deidda
# Author Kris Thielemans

# This file is part of STIR.
#
# SPDX-License-Identifier: Apache-2.0
#
# See STIR/LICENSE.txt for details


#%% imports

import stir
import stirextra
import pylab
import numpy
import math
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
#%% definition of useful objects and variables

scanner=stir.Scanner_get_scanner_from_name('SAFIRDualRingPrototype')
scanner.set_num_transaxial_blocks_per_bucket(2)
scanner.set_intrinsic_azimuthal_tilt(0)
# scanner.set_num_axial_crystals_per_block(1)
# scanner.set_axial_block_spacing(scanner.get_axial_crystal_spacing()*scanner.get_num_axial_crystals_per_block());
# scanner.set_num_rings(1)
scanner.set_scanner_geometry("BlocksOnCylindrical")

Nr=scanner.get_num_rings()
Nv=scanner.get_max_num_views()
NaBl=scanner.get_num_axial_blocks()
NtBu=scanner.get_num_transaxial_blocks()/scanner.get_num_transaxial_blocks_per_bucket()
aBl_s=scanner.get_axial_block_spacing()
tBl_s=scanner.get_transaxial_block_spacing()
tC_s=scanner.get_transaxial_crystal_spacing()
r=scanner.get_effective_ring_radius()

NtCpBl=scanner.get_num_transaxial_crystals_per_block()
NtBlpBu=scanner.get_num_transaxial_blocks_per_bucket()
NaCpBl=scanner.get_num_axial_crystals_per_block()
NaBlpBu=scanner.get_num_axial_blocks_per_bucket()
NCpR=scanner.get_num_detectors_per_ring()

min_r_diff=stir.IntVectorWithOffset((2*Nr-1))
max_r_diff=stir.IntVectorWithOffset((2*Nr-1))
num_ax_pps=stir.IntVectorWithOffset((2*Nr-1))

csi=math.pi/NtBu
tBl_gap=tBl_s-NtCpBl*tC_s
csi_minus_csiGaps=csi-(csi/tBl_s*2)*(tC_s/2+tBl_gap)
rmax =r/math.cos(csi_minus_csiGaps)

# scanner.set_intrinsic_azimuthal_tilt(-csi_minus_csiGaps) #if you want to play with the orientation of the blocks

#%% Create projection data info for Blocks on Cylindrical
for i in range(0,2*Nr-1,1 ):
    min_r_diff[i]=-Nr+1+i
    max_r_diff[i]=-Nr+1+i
    if i<Nr:
        num_ax_pps[i]=Nr+min_r_diff[i]
    else:
        num_ax_pps[i]=Nr-min_r_diff[i]
    # print(num_ax_pps[i])


proj_data_info_blocks=stir.ProjDataInfoBlocksOnCylindricalNoArcCorr(scanner, num_ax_pps, min_r_diff, max_r_diff, scanner.get_max_num_views(), scanner.get_max_num_non_arccorrected_bins())


#%% Plot 2D XY LORs for segment, axial position and tangential position =0
b1=stir.FloatCartesianCoordinate3D;
b2=stir.FloatCartesianCoordinate3D;
lor=stir.FloatLORInAxialAndNoArcCorrSinogramCoordinates;

fig=plt.figure()
ax=plt.axes()
plt.xlim([-rmax, rmax])
plt.ylim([-rmax, rmax])
ax.set_xlabel('X ')
ax.set_ylabel('Y ')

color_v = iter(cm.tab20(numpy.linspace(0, 10, NCpR)))
c=next(color_v)
tB_num2=-1

for v in range(0, Nv, 5):
    tB_nim_i, tB_num_f=divmod(v/NtCpBl,1)
    tB_num=int(tB_nim_i)
    bin=stir.Bin(0,v,0,0)

    if tB_num>tB_num2:
        c=next(color_v)

    tB_num2=tB_num
    b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
    b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
    
    plt.plot((b1.x(), b2.x()),(b1.y(), b2.y()),color=c)
    plt.plot(b1.x(),b1.y(),'o',color=c, label="Block %s - det %s" % (tB_num, v))

    # Shrink current axis %
    box = ax.get_position()
    ax.set_position([box.x0-box.y0*0.04, box.y0+box.y0*0.01, box.width, box.height])
    ax.set_aspect('equal', 'box')
    plt.legend(loc='best',bbox_to_anchor=(1., 1.),fancybox=True)
    # plt.show()  #if debugging we can se how the LORs are order
plt.gca().invert_yaxis()
plt.text(-65,65,"PL")
plt.text(65,65,"PR")
plt.text(-65,-65,"AL")
plt.text(65,-65,"AR")
plt.savefig('2D-2BlocksPerBuckets-ObliqueAt0degrees-XY-LOR.png', format='png', dpi=300)
plt.show()    
plt.close();

# # %% the following is an example when using LOR coordinates
# fig=plt.figure(figsize=(12, 12))
# ax=plt.axes()
# plt.xlim([-rmax, rmax])
# plt.ylim([-rmax, rmax])
# ax.set_xlabel('X ')
# ax.set_ylabel('Y ')

# for v in range(0, Nv, 5):
#     bin=stir.Bin(0,v,0,0)
#     lor=proj_data_info_blocks.get_lor(bin)
#     phi=lor.phi();
#     r=lor.radius()
#     plt.plot((r*math.sin(phi), r*math.sin(phi+math.pi)),(-r*math.cos(phi), -r*math.cos(phi+math.pi)))
#     plt.show()
# plt.gca().invert_yaxis()
# plt.show()    
# plt.savefig('2D-XY-LOR-cyl.png', format='png', dpi=300)

# plt.close()

#%% Plot 2D ZY LORs 
ax=plt.axes()
plt.xlim([0, NaBl*aBl_s])
plt.ylim([-rmax, rmax])
ax.set_xlabel('Z ')
ax.set_ylabel('Y ')

color_a = iter(cm.tab20(numpy.linspace(0, 1, Nr)))
c=next(color_a)
aB_num2=-1

for a in range(0,(Nr), 1):
    aB_nim_i, aB_num_f=divmod(a/NaCpBl,1)
    aB_num=int(aB_nim_i)
    bin=stir.Bin(0,v,0,0)

    if aB_num>aB_num2:
        c=next(color_a)

    aB_num2=aB_num
    bin=stir.Bin((Nr-1),0,a,0)
    b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
    b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
    
    plt.plot((b1.z(), b2.z()),(b1.y(), b2.y()),color=c)
    plt.plot(b1.z(),b1.y(),'o',color=c, label="Block %s - ring %s" % (aB_num, a))

    # Shrink current axis 
    box = ax.get_position()
    ax.set_position([box.x0-box.x0*0.01, box.y0+box.y0*0.01, box.width * .985, box.height])
    plt.legend(loc='best',bbox_to_anchor=(1., 1.),fancybox=True)

    # plt.show()
plt.gca().invert_yaxis()
plt.text(0.2,69,"PH")
plt.text(0.2,-65,"AH")
plt.text(34,69,"PF")
plt.text(34,-65,"AF")
plt.savefig('2D-YZ-LOR.png', format='png', dpi=300)
plt.show()
plt.close()


#%% Plot 3D  LORs 
ax=plt.axes(projection='3d')
ax.set_xlim([-rmax, rmax])
ax.set_ylim([-rmax, rmax])
ax.set_zlim([0, NaBl*aBl_s])
ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')
color_a = iter(cm.tab20(numpy.linspace(0, 1, Nr)))
c=next(color_a)
aB_num2=-1

for a in range(0,(Nr), 4):
    for v in range(0, Nv, 30):
        # aB_nim_i, aB_num_f=divmod(a/NaCpBl,1)
        # aB_num=int(aB_nim_i)
        bin=stir.Bin((Nr-1),v,a,0)
        
        # if aB_num>aB_num2:
        c=next(color_a)
            
        # aB_num2=aB_num
        b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
        b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
        plt.plot((b1.x(), b2.x()),(b1.y(), b2.y()),(b1.z(), b2.z()),color=c)
        ax.scatter3D(b1.x(),b1.y(),b1.z(),'o',color=c, label="ring %s - detector %s" % (a, v))

        # Shrink current axis by 1.5%
        box = ax.get_position()
        ax.set_position([box.x0-box.x0*0.12, box.y0+box.y0*0.01, box.width * .985, box.height])
        
        plt.legend(loc='best',bbox_to_anchor=(1., 1.),fancybox=True)
        # plt.show()
plt.gca().invert_yaxis()
plt.savefig('3dLOR.png', format='png', dpi=300)
plt.show()