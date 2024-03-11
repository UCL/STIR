/*
   Copyright (c) 2013, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain.
   Copyright (c) 2013, University College London
   This file is part of STIR.

   SPDX-License-Identifier: Apache-2.0

   See STIR/LICENSE.txt for details

 \author Carles Falcon

*/

#ifndef _WEIGHT3D_SPECTUB_H
#define _WEIGHT3D_SPECTUB_H

/*! \namespace SPECTUB
  \brief Namespace for the SPECT library from University of Barcelona

  This namespace encompasses function to construct a projection matrix suitable for SPECT.
  The integration of this library into STIR is described in

  Berta Marti Fuster, Carles Falcon, Charalampos Tsoumpas, Lefteris Livieratos, Pablo Aguiar, Albert Cot, Domenec Ros and Kris
  Thielemans, <i>Integration of advanced 3D SPECT modeling into the open-source STIR framework</i>, Med. Phys. 40, 092502 (2013);
  http://dx.doi.org/10.1118/1.4816676
*/

namespace SPECTUB
{

void wm_calculation(const int kOS,
                    const SPECTUB::angle_type* const ang,
                    SPECTUB::voxel_type vox,
                    bin_type bin,
                    const SPECTUB::volume_type& vol,
                    const proj_type& prj,
                    const float* attmap,
                    const bool* msk_3d,
                    const bool* msk_2d,
                    const int maxszb,
                    const SPECTUB::discrf_type* const gaussdens,
                    const int* const NITEMS,
                    SPECTUB::wm_da_type& wm,
                    SPECTUB::wmh_type& wmh,
                    const float* Rrad);

void wm_size_estimation(int kOS,
                        const SPECTUB::angle_type* const ang,
                        SPECTUB::voxel_type vox,
                        bin_type bin,
                        const SPECTUB::volume_type& vol,
                        const SPECTUB::proj_type& prj,
                        const bool* const msk_3d,
                        const bool* const msk_2d,
                        const int maxszb,
                        const SPECTUB::discrf_type* const gaussdens,
                        int* NITEMS,
                        SPECTUB::wmh_type& wmh,
                        const float* Rrad);

//... geometric component ............................................

void calc_gauss(SPECTUB::discrf_type* gaussdens);

void calc_vxprj(SPECTUB::angle_type* ang);

void voxel_projection(SPECTUB::voxel_type* vox, float* eff, float lngcmd2, SPECTUB::wmh_type& wmh);

void fill_psf_no(SPECTUB::psf2da_type* psf,
                 SPECTUB::psf1d_type* psf1d_h,
                 const SPECTUB::voxel_type& vox,
                 const angle_type* const ang,
                 float szdx,
                 SPECTUB::wmh_type& wmh);

void fill_psf_2d(SPECTUB::psf2da_type* psf,
                 SPECTUB::psf1d_type* psf1d_h,
                 const SPECTUB::voxel_type& vox,
                 SPECTUB::discrf_type const* const gaussdens,
                 float szdx,
                 SPECTUB::wmh_type& wmh);

void fill_psf_3d(SPECTUB::psf2da_type* psf,
                 SPECTUB::psf1d_type* psf1d_h,
                 SPECTUB::psf1d_type* psf1d_v,
                 const SPECTUB::voxel_type& vox,
                 SPECTUB::discrf_type const* const gaussdens,
                 float szdx,
                 float thdx,
                 float thcmd2,
                 wmh_type& wmh);

void calc_psf_bin(
    float center_psf, float binszcm, SPECTUB::discrf_type const* const vxprj, SPECTUB::psf1d_type* psf, SPECTUB::wmh_type& wmh);

//... attenuation...................................................

// not used
// void size_attpth_simple( psf2da_type *psf, voxel_type vox, volume_type vol, float *att, angle_type *ang );

// void size_attpth_full( psf2da_type *psf, voxel_type vox, volume_type vol, float *att, angle_type *ang );

void
calc_att_path(const bin_type& bin, const SPECTUB::voxel_type& vox, const SPECTUB::volume_type& vol, SPECTUB::attpth_type* attpth);

float calc_att(const SPECTUB::attpth_type* const attpth, const float* const attmap, int islc, SPECTUB::wmh_type& wmh);

int comp_dist(float dx, float dy, float dz, float dlast);

void error_weight3d(int nerr, const std::string& txt); // error messages in weight3d_SPECT

} // namespace SPECTUB

#endif
