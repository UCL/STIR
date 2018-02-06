 /*
    Copyright (c) 2013, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. 
    Copyright (c) 2013, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details

  \author Carles Falcon

*/

#ifndef _WEIGHT3D_SPECTUB_H
#define _WEIGHT3D_SPECTUB_H

/*! \namespace SPECTUB
  \brief Namespace for the SPECT library from University of Barcelona

  This namespace encompasses function to construct a projection matrix suitable for SPECT.
  The integration of this library into STIR is described in

  Berta Marti Fuster, Carles Falcon, Charalampos Tsoumpas, Lefteris Livieratos, Pablo Aguiar, Albert Cot, Domenec Ros and Kris Thielemans,
  <i>Integration of advanced 3D SPECT modeling into the open-source STIR framework</i>,
  Med. Phys. 40, 092502 (2013); http://dx.doi.org/10.1118/1.4816676

  \todo Variables wm, wmh and Rrad are currently global variables. This means that this code would be very dangerous
  in a parallel setting.
*/

namespace SPECTUB {

/* Global variables: the matrix etc TODO 
   Need to be defined elsewhere

   A lot of the functions below modify these variables.
*/
  extern wm_da_type wm; //! weight (or probability) matrix
  extern wmh_type wmh;  //! information to construct wm 
  extern float * Rrad;  //! radii per view


void wm_calculation( const int kOS,
					const angle_type *const ang, 
					voxel_type vox, 
					bin_type bin, 
					const volume_type& vol, 
					const proj_type& prj, 
					const float *attmap,
					const bool *msk_3d,
					const bool *msk_2d,
					const int maxszb,
					const discrf_type *const gaussdens,
					const int *const  NITEMS
					);

void wm_size_estimation (int kOS,
						 const angle_type * const ang, 
						 voxel_type vox, 
						 bin_type bin, 
						 const volume_type& vol, 
						 const proj_type& prj, 
						 const bool * const msk_3d,
						 const bool *const msk_2d,
						 const int maxszb,
						 const discrf_type * const gaussdens,
						 int *NITEMS);


//... geometric component ............................................

void calc_gauss ( discrf_type *gaussdens );             

void calc_vxprj ( angle_type *ang );

void voxel_projection ( voxel_type *vox, float * eff, float lngcmd2 );

void fill_psf_no( psf2da_type *psf, psf1d_type *psf1d_h, const voxel_type& vox, const angle_type * const ang, float szdx );

void fill_psf_2d( psf2da_type *psf, psf1d_type *psf1d_h, const voxel_type& vox, discrf_type const*const gaussdens, float szdx );

void fill_psf_3d(psf2da_type *psf,
                 psf1d_type *psf1d_h,
                 psf1d_type *psf1d_v,
                 const voxel_type& vox, discrf_type const * const gaussdens, float szdx, float thdx, float thcmd2 );

void calc_psf_bin ( float center_psf, float binszcm, discrf_type const * const vxprj, psf1d_type *psf );


//... attenuation...................................................

// not used
//void size_attpth_simple( psf2da_type *psf, voxel_type vox, volume_type vol, float *att, angle_type *ang );

//void size_attpth_full( psf2da_type *psf, voxel_type vox, volume_type vol, float *att, angle_type *ang );

void calc_att_path ( const bin_type& bin, const voxel_type& vox, const volume_type& vol, attpth_type *attpth );

float calc_att ( const attpth_type *const attpth, const float *const attmap, int islc );

int comp_dist ( float dx, float dy, float dz, float dlast );


void error_weight3d ( int nerr, const std::string& txt);               // error messages in weight3d_SPECT

} // napespace SPECTUB

#endif
