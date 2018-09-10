/*!
  \file
  \ingroup projection

  \brief stir::ProjMatrixByBinUsingRayTracing's definition 

  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
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
#ifndef __ProjMatrixByBinUsingRayTracing__
#define __ProjMatrixByBinUsingRayTracing__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ProjDataInfo;

/*!
  \ingroup projection
  \brief Computes projection matrix elements for VoxelsOnCartesianGrid images
  by using a Length of Intersection (LOI) model. 

  Currently, the LOIs are divided by voxel_size.x(), unless NEWSCALE is
  #defined during compilation time of ProjMatrixByBinUsingRayTracing.cxx. 

  It is possible to use multiple LORs in tangential direction. The result
  will then be the average of the various contributions. Currently all these
  LORs are parallel. For a very high number of LORs, the result approximates
  a strip integral (in tangential direction).

  If the z voxel size is exactly twice the sampling in axial direction,
  2 or 3 LORs are used, to avoid missing voxels. (TODOdoc describe how).

  If use_actual_detector_boundaries is set (currently only possible
  for non-arccorrected data, without mashing and/or axial compression), 
  the detectors are assumed to be on a cylinder. If only a single LOR 
  in tangential direction is used for ray tracing, 
  the centre of those detectors is used, which is slightly different from
  the 'usual' LOR (due to interleaving of the sinogram). When multiple
  LORs are used, the actual detector sizes are used, such that the resulting
  strip is twice as wide.

  It is possible to use a cylindrical or cuboid FOV (in the latter case it
  is going to be square in transaxial direction). In both cases, the FOV is 
  slightly 'inside' the image (i.e. it is about 1 voxel at each side
  smaller than the maximum possible).

  \par Symmetries

  For the azimuthal angle phi, the following angles are symmetry related for a square grid:
      {phi, 180-phi, 90-phi, 90+phi}.
  The boolean parameters &quot;do symmetry 90degrees min phi&quot;, &quot;do symmetry 180degrees min phi&quot;
  allow to select if all 4 angles should be considered as related.

  Enabling more symmetries (from DataSymmetriesForBins_PET_CartesianGrid) means that less memory is
  needed to store the matrix (when caching), less time to compute it, but using the matrix might be
  slightly slower. By default, as many symmetries as possible are enabled.
  
  \par Parsing parameters

  The following parameters can be set (default values are indicated):
  \verbatim
  Ray Tracing Matrix Parameters :=
  ; any parameters appropriate for class ProjMatrixByBin
  restrict to cylindrical FOV := 1
  number of rays in tangential direction to trace for each bin := 1
  use actual detector boundaries := 0
  do symmetry 90degrees min phi := 1
  do symmetry 180degrees min phi := 1
  do_symmetry_swap_segment := 1
  do_symmetry_swap_s := 1
  do_symmetry_shift_z := 1
  End Ray Tracing Matrix Parameters :=
  \endverbatim
                  
  \par Implementation details

  The implementation uses RayTraceVoxelsOnCartesianGrid().

  \warning After calling any of the \c set functions or parsing, you have to call setup(), otherwise
  using the matrix will result in a call to error().

  \warning Only appropriate for VoxelsOnCartesianGrid type of images
  (otherwise error() will be called).
  
  \warning Care should be taken to select the number of rays in tangential direction 
  such that the sampling is at least as small as the x,y voxel sizes.
  \warning Current implementation assumes that z voxel size is either
  smaller than or exactly twice the sampling in axial direction of the segments.

  \bug Currently, strange things happen if the z voxel size is not exactly equal to half 
  the ring spacing of the scanner.
*/

class ProjMatrixByBinUsingRayTracing : 
  public RegisteredParsingObject<
	      ProjMatrixByBinUsingRayTracing,
              ProjMatrixByBin,
              ProjMatrixByBin
	       >
{
public :
    //! Name which will be used when parsing a ProjMatrixByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  ProjMatrixByBinUsingRayTracing();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  //! \name If a cylindrical FOV or the whole image will be handled
  //!@{
  bool get_restrict_to_cylindrical_FOV() const;
  void set_restrict_to_cylindrical_FOV(bool);
  //!@}
  //! \name How many rays will be traced in tangential direction for one bin
  //!@{
  int get_num_tangential_LORs() const;
  void set_num_tangential_LORs(int);
  //!@}

  //! \name If interleaved sinogram coordinates are used or not.
  //!@{
  bool get_use_actual_detector_boundaries() const;
  void set_use_actual_detector_boundaries(bool);
  //@}

  //! \name Which symmetries will be used
  //!@{
  bool get_do_symmetry_90degrees_min_phi() const;
  void set_do_symmetry_90degrees_min_phi(bool);

  bool get_do_symmetry_180degrees_min_phi() const;
  void set_do_symmetry_180degrees_min_phi(bool);

  bool get_do_symmetry_swap_segment() const;
  void set_do_symmetry_swap_segment(bool);

  bool get_do_symmetry_swap_s() const;
  void set_do_symmetry_swap_s(bool);

  bool get_do_symmetry_shift_z() const;
  void set_do_symmetry_shift_z(bool);
  //!@}

private:
  //! variable to keep track if setup is called already
  /*! Using any of the \c set function will set it to false, so you will have to call setup() again.
   */
  bool already_setup;

  //! variable that determines if a cylindrical FOV or the whole image will be handled
  bool restrict_to_cylindrical_FOV;
  //! variable that determines how many rays will be traced in tangential direction for one bin
  int num_tangential_LORs;
  //! variable that determines if interleaved sinogram coordinates are used or not.
  bool use_actual_detector_boundaries;
  bool do_symmetry_90degrees_min_phi;
  bool do_symmetry_180degrees_min_phi;
  bool do_symmetry_swap_segment;
  bool do_symmetry_swap_s;
  bool do_symmetry_shift_z;

  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  shared_ptr<ProjDataInfo> proj_data_info_sptr;
  shared_ptr< DiscretisedDensity<3,float> > density_info_sptr;


  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

   virtual void set_defaults();
   virtual void initialise_keymap();
   virtual bool post_processing();
  
};

END_NAMESPACE_STIR

#endif



