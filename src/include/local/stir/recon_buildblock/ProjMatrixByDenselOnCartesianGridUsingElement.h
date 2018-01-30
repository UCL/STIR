//
//
/*!
  \file
  \ingroup recon_buildblock

  \brief ProjMatrixByDenselOnCartesianGridUsingElement's definition 

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjMatrixByDenselOnCartesianGridUsingElement__
#define __ProjMatrixByDenselOnCartesianGridUsingElement__

#include "local/stir/recon_buildblock/ProjMatrixByDensel.h"
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
  #defined during compilation time of ProjMatrixByDenselOnCartesianGridUsingElement.cxx. 

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

class ProjMatrixByDenselOnCartesianGridUsingElement : 
  public ProjMatrixByDensel
{
public :

  //! Stores necessary geometric info
  /*! This function \c hsd to be called by any derived class.

      Note that the density_info_ptr is not stored in this object. It's only 
      used to get some info on sizes etc.

      Currently, the proj_data_info_ptr argument is not used.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  //! this member computes a single element of the projection matrix
  /*! \param bin The bin-coordinates specifying the row of the projection matrix
      \param densel_ctr The densel coordinates specifying the column of the projection matrix
      \return returns the value of the element

      Ideally, get_element should be implemented such that tiny
      elements are truncated to 0, to avoid storing (and using)
      elements that not really contribute to the result.
      
      \warning Currently, the densel_ctr has to be in mm and w.r.t. the
               centre of the scanner. This is a bad idea (it should use
               dicrete coordinates) and so will change in the future.
               It's there now for the RT projector to avoid recomputing
               the coordinates per mm fo revery bin.
*/
  virtual float 
     get_element(const Bin& bin,
                 const CartesianCoordinate3D<float>& densel_ctr) const = 0;
protected:
  shared_ptr<ProjDataInfo> proj_data_info_ptr;

  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<float> grid_spacing;
  CartesianCoordinate3D<float> origin;  
  float min_z_index;
  float max_z_index;

  //! Calculates all non-zero elements for a particular densel
  /*! This implementation uses the get_element() member. It uses a generic
      way of finding non-zero elements, which is slow but makes only a 
      few assumptions.
      TODO more doc.
  */
  void calculate_proj_matrix_elems_for_one_densel(ProjMatrixElemsForOneDensel &) const;
  
  
};

END_NAMESPACE_STIR

#endif



