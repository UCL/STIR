//
// $Id$
//
/*!

  \file
  \ingroup projection
  
  \brief Declaration of class stir::ForwardProjectorByBinUsingRayTracing
    
  \author Kris Thielemans
  \author PARAPET project
      
   $Date$
   $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_ForwardProjectorByBinUsingRayTracing__H__
#define __stir_recon_buildblock_ForwardProjectorByBinUsingRayTracing__H__

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/shared_ptr.h"
START_NAMESPACE_STIR

template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <typename elemT> class VoxelsOnCartesianGrid;
template <int num_dimensions, typename elemT> class Array;
class ProjDataInfo;
class ProjDataInfoCylindrical;


/*!
  \ingroup projection
  \brief This class implements forward projection using Siddon's algorithm for
  ray tracing. That is, it computes length of intersection with the voxels.

  Currently, the LOIs are divided by voxel_size.x(), unless NEWSCALE is
  #defined during compilation time of ForwardProjectorByBinUsingRayTracing_Siddon.cxx. 

  If the z voxel size is exactly twice the sampling in axial direction,
  multiple LORs are used, to avoid missing voxels. (TODOdoc describe how).

  Currently, a FOV is used which is circular, and is slightly 'inside' the 
  image (i.e. the radius is about 1 voxel smaller than the maximum possible).

  \warning Current implementation assumes that x,y voxel sizes are at least as 
  large as the sampling in tangential direction, and that z voxel size is either
  equal to or exactly twice the sampling in axial direction of the segments.

  \warning For each bin, maximum 3 LORs are 'traced'
  \warning The image forward projected HAS to be of type VoxelsOnCartesianGrid.
  \warning The projection data info HAS to be of type ProjDataInfoCylindrical
  \warning The implementation assumes that the \c s -coordinate is antisymmetric
  in terms of the tangential_pos_num, i.e.
  \code
  proj_data_info_ptr->get_s(Bin(...,tang_pos_num)) == 
  - proj_data_info_ptr->get_s(Bin(...,-tang_pos_num))
  \endcode
*/

class ForwardProjectorByBinUsingRayTracing : 
  public RegisteredParsingObject<ForwardProjectorByBinUsingRayTracing,
                                 ForwardProjectorByBin>
{
public:
    //! Name which will be used when parsing a ForwardProjectorByBin object
  static const char * const registered_name; 


  ForwardProjectorByBinUsingRayTracing();

  //! Constructor
  /*! \warning Obsolete */
  ForwardProjectorByBinUsingRayTracing(
                       const shared_ptr<ProjDataInfo>&,
                       const shared_ptr<DiscretisedDensity<3,float> >&);
  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  virtual const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;

 protected:
  //! variable that determines if a cylindrical FOV or the whole image will be handled
  bool restrict_to_cylindrical_FOV;


private:
  void actual_forward_project(RelatedViewgrams<float>&, 
		  const DiscretisedDensity<3,float>&,
		  const int min_axial_pos_num, const int max_axial_pos_num,
		  const int min_tangential_pos_num, const int max_tangential_pos_num);


  // KT 20/06/2001 changed type from 'const DataSymmetriesForViewSegmentNumbers *'
  shared_ptr<DataSymmetriesForBins_PET_CartesianGrid> symmetries_ptr;
  /*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
    */

  void 
  forward_project_all_symmetries(
				Viewgram<float> & pos_view, 
				 Viewgram<float> & neg_view, 
				 Viewgram<float> & pos_plus90, 
				 Viewgram<float> & neg_plus90, 
				 Viewgram<float> & pos_min180, 
				 Viewgram<float> & neg_min180, 
				 Viewgram<float> & pos_min90, 
				 Viewgram<float> & neg_min90, 
				 const VoxelsOnCartesianGrid<float>& image,
				 const int min_axial_pos_num, const int max_axial_pos_num,
				 const int min_tangential_pos_num, const int max_tangential_pos_num) const;


  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees 
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void 
  forward_project_view_plus_90_and_delta(
					 Viewgram<float> & pos_view, 
					 Viewgram<float> & neg_view, 
					 Viewgram<float> & pos_plus90, 
					 Viewgram<float> & neg_plus90, 
					 const VoxelsOnCartesianGrid<float> & image,
					 const int min_axial_pos_num, const int max_axial_pos_num,
					 const int min_tangential_pos_num, const int max_tangential_pos_num) const; 
  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees 
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void 
  forward_project_view_min_180_and_delta(
					 Viewgram<float> & pos_view, 
					 Viewgram<float> & neg_view, 
					 Viewgram<float> & pos_min180, 
					 Viewgram<float> & neg_min180, 
					 const VoxelsOnCartesianGrid<float> & image,
					 const int min_axial_pos_num, const int max_axial_pos_num,
					 const int min_tangential_pos_num, const int max_tangential_pos_num) const; 

  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees 
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void 
  forward_project_delta(
			    Viewgram<float> & pos_view, 
			    Viewgram<float> & neg_view, 
			    const VoxelsOnCartesianGrid<float> & image,
			    const int min_axial_pos_num, const int max_axial_pos_num,
			    const int min_tangential_pos_num, const int max_tangential_pos_num) const; 

  //////////////// 2D 
  void forward_project_all_symmetries_2D(
					 Viewgram<float> & pos_view, 
					 Viewgram<float> & pos_plus90, 
					 Viewgram<float> & pos_min180, 
					 Viewgram<float> & pos_min90, 
					 const VoxelsOnCartesianGrid<float>& image,
					 const int min_axial_pos_num, const int max_axial_pos_num,
					 const int min_tangential_pos_num, const int max_tangential_pos_num) const;
  void 
forward_project_view_plus_90_2D(Viewgram<float> & pos_view, 
				Viewgram<float> & pos_plus90, 
				const VoxelsOnCartesianGrid<float> & image,
				const int min_axial_pos_num, const int max_axial_pos_num,
				const int min_tangential_pos_num, const int max_tangential_pos_num) const;
void 
forward_project_view_min_180_2D(Viewgram<float> & pos_view, 
			       Viewgram<float> & pos_min180, 
			       const VoxelsOnCartesianGrid<float> & image,
			       const int min_axial_pos_num, const int max_axial_pos_num,
			       const int min_tangential_pos_num, const int max_tangential_pos_num) const;
// no symmetries
void 
forward_project_view_2D(Viewgram<float> & pos_view, 
				          const VoxelsOnCartesianGrid<float> & image,
 				          const int min_axial_pos_num, const int max_axial_pos_num,
				          const int min_tangential_pos_num, const int max_tangential_pos_num) const;
#if defined(_MSC_VER) && _MSC_VER<1310
/* VC 6.0 (and 7.0 ?) cannot use the normal syntax unfortunately
   See also http://www.boost.org/more/microsoft_vcpp.html

   So, we forget about the template in this case.
   Sigh
*/
#define STIR_SIDDON_NO_TEMPLATE
#endif

#ifndef STIR_SIDDON_NO_TEMPLATE
  //! The actual implementation of Siddon's algorithm 
  /*! \return true if the LOR intersected the image, i.e. of Projptr (potentially) changed */ 
  template <int symmetry_type> 
  static bool 
    proj_Siddon(Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
			  const ProjDataInfoCylindrical* proj_data_info_ptr, 
			  const float cphi, const float sphi, const float delta, 
			  const float s_in_mm, 
			  const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
			  const int num_planes_per_axial_pos,
			  const float axial_pos_to_z_offset,
			  const float norm_factor,
			  const bool restrict_to_cylindrical_FOV);
#else
  static bool
    proj_Siddon(int symmetry_type,
                Array<4,float> &Projptr, const VoxelsOnCartesianGrid<float> &, 
			  const ProjDataInfoCylindrical* proj_data_info_ptr, 
			  const float cphi, const float sphi, const float delta, 
			  const float s_in_mm, 
			  const float R, const int min_ax_pos_num, const int max_ax_pos_num, const float offset, 
			  const int num_planes_per_axial_pos,
			  const float axial_pos_to_z_offset,
			  const float norm_factor,
			  const bool restrict_to_cylindrical_FOV);
#endif

  virtual void set_defaults();
  virtual void initialise_keymap();
};
END_NAMESPACE_STIR
#endif
