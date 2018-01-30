//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByDenselUsingRayTracing's definition 

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjMatrixByDenselUsingRayTracing__
#define __ProjMatrixByDenselUsingRayTracing__

#include "stir/RegisteredParsingObject.h"
#include "local/stir/recon_buildblock/ProjMatrixByDenselOnCartesianGridUsingElement.h"
#include "stir/CartesianCoordinate3D.h"

 

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
              ProjMatrixByDenselOnCartesianGridUsingElement
	       >
{
  typedef ProjMatrixByDenselOnCartesianGridUsingElement base_type;
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

  virtual float 
    get_element(const Bin&, const CartesianCoordinate3D<float>&) const;

private:
  shared_ptr<DataSymmetriesForDensels_PET_CartesianGrid> symmetries_ptr;
#if 0
  //! variable that determines if a cylindrical FOV or the whole image will be handled
  bool restrict_to_cylindrical_FOV;
#endif
  //! variable that determines how many rays will be traced in tangential direction for one bin
  int num_tangential_LORs;
  //! variable that determines if interleaved sinogram coordinates are used or not.
  bool use_actual_detector_boundaries;  

  // explicitly list necessary members for image details (should use an Info object instead)
  // ideally these should be const, but I have some trouble initialising them in that case
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  CartesianCoordinate3D<int> min_index;
  CartesianCoordinate3D<int> max_index;

  float xhalfsize;
  float yhalfsize;
  float zhalfsize;



   virtual void set_defaults();
   virtual void initialise_keymap();
   virtual bool post_processing();
  
};

END_NAMESPACE_STIR

#endif



