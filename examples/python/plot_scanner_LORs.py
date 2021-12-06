# Demo of how to use STIR from python to reconstruct some data
# To run in "normal" Python, you would type the following in the command line
#  execfile('recon_demo.py')
# In ipython, you can use
#  %run recon_demo.py

# Copyright 2012-06-05 - 2013 Kris Thielemans
# Copyright 2015 University College London

# This file is part of STIR.
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
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

#%% definition of useful objects and variables

scanner=stir.Scanner_get_scanner_from_name('SAFIRDualRingPrototype')
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
NCpR=scanner.get_num_detectors_per_ring()

min_r_diff=stir.IntVectorWithOffset((2*Nr-1))
max_r_diff=stir.IntVectorWithOffset((2*Nr-1))
num_ax_pps=stir.IntVectorWithOffset((2*Nr-1))

csi=math.pi/NtBu
tBl_gap=tBl_s-NtCpBl*tC_s
csi_minus_csiGaps=csi-(csi/tBl_s*2)*(tC_s/2+tBl_gap)
rmax =r/math.cos(csi_minus_csiGaps)


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
lor=stir.FloatLOR;

fig=plt.figure(figsize=(12, 12))
ax=plt.axes()
plt.xlim([-rmax, rmax])
plt.ylim([-rmax, rmax])
ax.set_xlabel('X ')
ax.set_ylabel('Y ')

for v in range(0, Nv, 5):
    bin=stir.Bin(0,v,0,0)
    b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
    b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
    plt.plot((b1.x(), b2.x()),(b1.y(), b2.y()))
    # plt.show()  #if debugging we can se how the LORs are order
plt.show()    
plt.savefig('2D-XY-LOR.png', format='png', dpi=300)

plt.close()

# for v in range(0, Nv, 5):
#     bin=stir.Bin(0,v,0,0)
#     lor=proj_data_info_blocks.get_lor(bin)
#     phi=lor.phi();
#     plt.plot((b1.x(), b2.x()),(b1.y(), b2.y()))
#     # plt.show()
# plt.show()    
# plt.savefig('2D-XY-LOR.png', format='png', dpi=300)

# plt.close()

#%% Plot 2D ZY LORs 
ax=plt.axes()
plt.xlim([0, NaBl*aBl_s])
plt.ylim([-rmax, rmax])
ax.set_xlabel('Z ')
ax.set_ylabel('Y ')
for a in range(0,(Nr-1), 1):
    bin=stir.Bin((Nr-1),0,a,0)
    b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
    b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
    plt.plot((b1.z(), b2.z()),(b1.y(), b2.y()))
    # plt.show()
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
for a in range(0,(Nr-1), 1):
    for v in range(0, Nv, 15):
        bin=stir.Bin((Nr-1),v,a,0)
        b1=proj_data_info_blocks.find_cartesian_coordinate_of_detection_1(bin)
        b2=proj_data_info_blocks.find_cartesian_coordinate_of_detection_2(bin)
        plt.plot((b1.x(), b2.x()),(b1.y(), b2.y()),(b1.z(), b2.z()))
        plt.show()

plt.savefig('3dLOR.png', format='png', dpi=300)
