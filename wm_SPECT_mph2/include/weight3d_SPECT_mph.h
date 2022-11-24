/*
 * Copyright (c) 2014, 
 * Institute of Nuclear Medicine, University College of London Hospital, UCL, London, UK.
 * Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
 * This software is distributed WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 
 \author Carles Falcon
 */

#ifndef _WEIGHT3D_SPECT_mph_H
#define _WEIGHT3D_SPECT_mph_H

namespace SPECTUB_mph
{

  //... global variables ..............................................

  extern wm_da_type wm;
  extern wmh_mph_type wmh;
  extern pcf_type pcf;

  void wm_calculation_mph ( bool do_estim,
			    psf2d_type *psf2d_bin,
			    psf2d_type *psf_subs ,
			    psf2d_type *psf2d_aux ,
			    psf2d_type *kern,
			    float *attmap,
			    bool *msk_3d,
			    int *Nitems );

  //... geometric component ............................................

  bool check_xang_par( voxel_type * vox, hole_type * h);

  bool check_zang_par( voxel_type * vox, hole_type * h);

  //bool check_xang_obl( lor_type * l, voxel_type * vox, hole_type * h);
  //
  //bool check_zang_obl( lor_type * l, voxel_type * vox, hole_type * h);

  void voxel_projection_mph ( lor_type * l, voxel_type * v, hole_type * h );


  void fill_psfi ( psf2d_type * kern );

  void downsample_psf ( psf2d_type * psf_in, psf2d_type * psf_out, int factor, bool do_calc );

  void psf_convol( psf2d_type * psf1, psf2d_type * psf_aux, psf2d_type * psf2, bool do_calc );

  float bresenh_f( int i1, int j1, int i2, int j2, float ** f, int imax, int jmax, float dcr );


  void fill_psf_geo ( psf2d_type *psf2d, lor_type *l, discrf2d_type *f, int factor, bool do_calc );

  void fill_psf_depth ( psf2d_type *psf2d, lor_type *l, discrf2d_type *f, int factor, bool do_calc );

  void psf_convol ( psf2d_type * psf2d, psf2d_type * psf_aux, psf2d_type * kern );

  void downsample_psf ( psf2d_type * psf_subs, psf2d_type * psf_bin );

  //... attenuation...................................................

  float calc_att_mph ( bin_type bin, voxel_type vox, float *attmap );

  int comp_dist ( float dx, float dy, float dz, float dlast );


  void error_weight3d ( int nerr, std::string txt);               // error messages in weight3d_SPECT

}

#endif
