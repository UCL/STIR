//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByDenselUsingRayTracing's definition 

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjMatrixByDenselUsingRayTracing__
#define __ProjMatrixByDenselUsingRayTracing__

#include "stir/RegisteredParsingObject.h"
#include "local/stir/recon_buildblock/ProjMatrixByDensel.h"
#include "stir/ProjDataInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class DataSymmetriesForDensels_PET_CartesianGrid;

/*!
  \ingroup recon_buildblock
  \brief Computes projection matrix elements for VoxelsOnCartesianGrid images
  by using a Length of Intersection (LOI) model. 

  Currently, the LOIs are divided by voxel_size.x(), unless NEWSCALE is
  #defined during compilation time of ProjMatrixByDenselUsingRayTracing.cxx. 

  If the z voxel size is exactly twice the sampling in axial direction,
  multiple LORs are used, to avoid missing voxels. (TODOdoc describe how).

  Currently, a FOV is used which is circular, and is slightly 'inside' the 
  image (i.e. the radius is about 1 voxel smaller than the maximum possible).

  The implementation uses RayTraceVoxelsOnCartesianGrid().

  \warning Only appropriate for VoxelsOnCartesianGrid type of images
  (otherwise a run-time error occurs).
  
  \warning Current implementation assumes that x,y voxel sizes are at least as 
  large as the sampling in tangential direction, and that z voxel size is either
  smaller than or exactly twice the sampling in axial direction of the segments.

*/

class ProjMatrixByDenselUsingRayTracing : 
  public RegisteredParsingObject<
	      ProjMatrixByDenselUsingRayTracing,
              ProjMatrixByDensel
	       >
{
public :
    //! Name which will be used when parsing a ProjMatrixByDensel object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  ProjMatrixByDenselUsingRayTracing();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  virtual const  DataSymmetriesForDensels* get_symmetries_ptr() const;

private:
  shared_ptr<DataSymmetriesForDensels_PET_CartesianGrid> symmetries_ptr;
  
  // explicitly list necessary members for image details (should use an Info object instead)
  // ideally these should be const, but I have some trouble initialising them in that case
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  float xhalfsize;
  float yhalfsize;
  float zhalfsize;

  shared_ptr<ProjDataInfo> proj_data_info_ptr;


  virtual void 
    calculate_proj_matrix_elems_for_one_densel(
                                            ProjMatrixElemsForOneDensel&) const;

   virtual void set_defaults();
   virtual void initialise_keymap();
  
};

END_NAMESPACE_STIR

#endif



