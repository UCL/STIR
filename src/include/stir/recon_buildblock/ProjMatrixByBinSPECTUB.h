/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
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
*/
/*!
  \file
  \ingroup projection

  \brief stir::ProjMatrixByBinSPECTUB's definition 

  \author Berta Marti Fuster
  \author Kris Thielemans
*/
#ifndef __stir_recon_buildblock_ProjMatrixByBinSPECTUB__
#define __stir_recon_buildblock_ProjMatrixByBinSPECTUB__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"
#include <iostream>


#include "stir/recon_buildblock/SPECTUB_Tools.h"
#ifdef abs
   #undef abs // TODO modify WMTools.h
#endif

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Bin;
/*!
  \ingroup projection
  \brief generates projection matrix for SPECT studies


  \todo this class currently only works with VoxelsOnCartesianGrid. 
*/

class ProjMatrixByBinSPECTUB : 
  public RegisteredParsingObject<
	      ProjMatrixByBinSPECTUB,
              ProjMatrixByBin,
              ProjMatrixByBin
	       >
{
 public :
  //! Name which will be used when parsing a ProjMatrixByBin object
  static const char * const registered_name; 
  
  //! Default constructor (calls set_defaults())
  ProjMatrixByBinSPECTUB();

  //! Destructor (deallocates UB SPECT memory)
  ~ProjMatrixByBinSPECTUB();

  //! Checks all necessary geometric info
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
                      );

 private:

 
  float minimum_weight;
  float maximum_number_of_sigmas;
  float spatial_resolution_PSF;
  string psf_type;
  float collimator_sigma_0;
  float collimator_slope;
  string attenuation_type;
  string attenuation_map;
  string mask_type;
  string mask_file;

  // TODO this only works as long as we only have VoxelsOnCartesianGrid
  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  IndexRange<3> densel_range;

  shared_ptr<ProjDataInfo> proj_data_info_ptr;

  bool already_setup;


  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  // wm_SPECT starts here ---------------------------------------------------------------------------------------------
  bool *msk_3d, *msk_2d; // voxels to be included in matrix (no weight calculated outside the mask) and its 2d collapse

  //... variables for estimated sizes of arrays to allocate ................................
  int **NITEMS;           // number of non-zero elements for each weight matrix row

  //... user defined structures (defined in wm_SPECT.h) .....................................

  volume_type vol;       // structure with volume (image) information
  proj_type prj;         // structure with projection information

  voxel_type vox;        // structure with voxel information
  bin_type bin;          // structure with bin information

  angle_type * ang;      // structure with angle indices, values, ratios and voxel projections
  float *attmap;         // attenuation map

  bool keep_all_views_in_cache;
	
  discrf_type gaussdens; // strucutre with gaussian density function 

  int maxszb;

  void compute_one_subset(const int kOS) const;
  void delete_UB_SPECT_arrays();
  mutable std::vector<bool> subset_already_processed;
};

END_NAMESPACE_STIR

#endif



