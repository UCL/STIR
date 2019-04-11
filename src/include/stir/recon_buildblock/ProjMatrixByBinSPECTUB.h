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

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Bin;
/*!
  \ingroup projection
  \brief generates projection matrix for SPECT studies

  This functionality is described in
  Berta Marti Fuster, Carles Falcon, Charalampos Tsoumpas, Lefteris Livieratos, Pablo Aguiar, Albert Cot, Domenec Ros and Kris Thielemans,
  <i>Integration of advanced 3D SPECT modeling into the open-source STIR framework</i>,
  Med. Phys. 40, 092502 (2013); http://dx.doi.org/10.1118/1.4816676

  \warning this class currently only works with VoxelsOnCartesianGrid. 

  \par Sample parameter file

\verbatim
  Projection Matrix By Bin SPECT UB Parameters:=				
    ; width of PSF
    maximum number of sigmas:= 2.0

    ;PSF type of correction { 2D // 3D // Geometrical }
    psf type:= 2D
    ; next 2 parameters define the PSF. They are ignored if psf_type is "Geometrical"
    ; the PSF is modelled as a Gaussian with sigma dependent on the distance from the collimator
    ; sigma_at_depth = collimator_slope * depth_in_cm + collimator sigma 0(cm)
    collimator slope := 0.0163
    collimator sigma 0(cm) := 0.1466

    ;Attenuation correction { Simple // Full // No }
    attenuation type := Simple	
    ;Values in attenuation map in cm-1			
    attenuation map := attMapRec.hv

    ;Mask properties { Cylinder // Attenuation Map // Explicit Mask // No}
    mask type := Explicit Mask
    mask file := mask.hv

    ; if next variable is set to 0, only a single view is kept in memory
   keep all views in cache:=1

End Projection Matrix By Bin SPECT UB Parameters:=
\endverbatim
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

  bool get_keep_all_views_in_cache() const;
  //! Enable keeping the matrix in memory
  /*!
    This speeds-up the calculations, but can use a lot of memory.

    You have to call set_up() after this (unless the value didn't change).
  */
  void set_keep_all_views_in_cache(bool value = true);
  std::string get_attenuation_type() const;
  //! Set type of attenuation modelling
  /* Has to be "no", "simple" or "full"

    You have to call set_up() after this (unless the value didn't change).
  */
  void set_attenuation_type(const std::string& value);
  shared_ptr<const DiscretisedDensity<3,float> >
    get_attenuation_image_sptr() const;
  //! Sets attenuation image
  /*!
    The image has to have same characteristics as the emission image currently.

    Will call set_attenuation_type() to set to "simple" if it was set to "no".

    You have to call set_up() after this.
  */
  void
    set_attenuation_image_sptr(const shared_ptr<const DiscretisedDensity<3,float> > value);
  void
    set_attenuation_image_sptr(const std::string& value);
  //! Set the parameters for the depth-dependent resolution model
  /*!
    The detector and collimator blurring is modelled as a Gaussian with sigma dependent on the
    distance from the collimator.

    \verbatim
     sigma_at_depth = collimator_slope * depth_in_mm + collimator sigma 0
    \end_verbatim

    Set slope and sigma_0 to zero for "geometrical" modelling.

    \warning Values are in mm

    You have to call set_up() after this.
  */

  void
    set_resolution_model(const float collimator_sigma_0_in_mm, const float collimator_slope_in_mm, const bool full_3D = true);

 private:

  // parameters that will be parsed
 
  float minimum_weight;
  float maximum_number_of_sigmas;
  float spatial_resolution_PSF;
  std::string psf_type;
  float collimator_sigma_0;
  float collimator_slope;
  std::string attenuation_type;
  std::string attenuation_map;
  std::string mask_type;
  std::string mask_file;
  bool keep_all_views_in_cache; //!< if set to false, only a single view is kept in memory

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

  shared_ptr<const DiscretisedDensity<3,float> > attenuation_image_sptr;

  // wm_SPECT starts here ---------------------------------------------------------------------------------------------
  bool *msk_3d; //!< voxels to be included in matrix (no weight calculated outside the mask) 
  bool *msk_2d; //!< 2d collapse of msk_3d.

  //... variables for estimated sizes of arrays to allocate ................................
  int **NITEMS;           //!< number of non-zero elements for each weight matrix row

  //... user defined structures (types defined in SPECTUB_Tools.h) .....................................

  SPECTUB::volume_type vol;       //!< structure with volume (image) information
  SPECTUB::proj_type prj;         //!< structure with projection information

  SPECTUB::voxel_type vox;        //!< structure with voxel information
  SPECTUB::bin_type bin;          //!< structure with bin information

  SPECTUB::angle_type * ang;      //!< structure with angle indices, values, ratios and voxel projections
  float *attmap;         //!< attenuation map (copied as float array)

  SPECTUB::discrf_type gaussdens; //!< structure with gaussian density function 

  int maxszb;

	
  void compute_one_subset(const int kOS) const;
  void delete_UB_SPECT_arrays();
  mutable std::vector<bool> subset_already_processed;
};

END_NAMESPACE_STIR

#endif



