//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByBinUsingRayTracing's definition 

  \author Kris Thielemans
  \author Mustapha Sadki
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjMatrixByBinUsingRayTracing__
#define __ProjMatrixByBinUsingRayTracing__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup recon_buildblock
  \brief Computes projection matrix elements for VoxelsOnCartesianGrid images
  by using a Length of Intersection (LOI) model. 

  Currently, the LOIs are divided by voxel_size.x(), unless NEWSCALE is
  #defined during compilation time of ProjMatrixByBinUsingRayTracing.cxx. 

  If the z voxel size is exactly twice the sampling in axial direction,
  multiple LORs are used, to avoid missing voxels. (TODOdoc describe how).

  It is possible to use a cylindrical or cuboid FOV (in the latter case it
  is going to be square in transaxial direction). In both cases, the FOV is 
  slightly 'inside' the image (i.e. it is about 1 voxel at each side
  smaller than the maximum possible).

  The implementation uses RayTraceVoxelsOnCartesianGrid().

  \warning Only appropriate for VoxelsOnCartesianGrid type of images
  (otherwise a run-time error occurs).
  
  \warning Current implementation assumes that x,y voxel sizes are at least as 
  large as the sampling in tangential direction, and that z voxel size is either
  smaller than or exactly twice the sampling in axial direction of the segments.

*/

class ProjMatrixByBinUsingRayTracing : 
  public RegisteredParsingObject<
	      ProjMatrixByBinUsingRayTracing,
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

private:
  //! variable that determines if a cylindrical FOV or the whole image will be handled
  bool restrict_to_cylindrical_FOV;

  // explicitly list necessary members for image details (should use an Info object instead)
  // ideally these should be const, but I have some trouble initialising them in that case
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  shared_ptr<ProjDataInfo> proj_data_info_ptr;


  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

   virtual void set_defaults();
   virtual void initialise_keymap();
  
};

END_NAMESPACE_STIR

#endif



