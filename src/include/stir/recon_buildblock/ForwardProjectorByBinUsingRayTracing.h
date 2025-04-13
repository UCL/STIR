//
//
/*!

  \file
  \ingroup projection

  \brief Declaration of class stir::ForwardProjectorByBinUsingRayTracing

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_ForwardProjectorByBinUsingRayTracing__H__
#define __stir_recon_buildblock_ForwardProjectorByBinUsingRayTracing__H__

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/ArrayFwd.h"

START_NAMESPACE_STIR

template <typename elemT>
class Viewgram;
template <typename elemT>
class RelatedViewgrams;
template <typename elemT>
class VoxelsOnCartesianGrid;
class ProjDataInfo;
class ProjDataInfoCylindrical;

/*!
  \ingroup projection
  \brief This class implements forward projection using Siddon's algorithm for
  ray tracing. That is, it computes length of intersection with the voxels.

  Currently, the LOIs are divided by voxel_size.x(), unless \c NEWSCALE is
  \c \#defined during compilation time of ForwardProjectorByBinUsingRayTracing_Siddon.cxx.

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

class ForwardProjectorByBinUsingRayTracing
    : public RegisteredParsingObject<ForwardProjectorByBinUsingRayTracing, ForwardProjectorByBin>
{
public:
  //! Name which will be used when parsing a ForwardProjectorByBin object
  static const char* const registered_name;

  ForwardProjectorByBinUsingRayTracing();

  //! Constructor
  /*! \warning Obsolete */
  ForwardProjectorByBinUsingRayTracing(const shared_ptr<const ProjDataInfo>&,
                                       const shared_ptr<const DiscretisedDensity<3, float>>&);
  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
   */
  void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
              const shared_ptr<const DiscretisedDensity<3, float>>& density_info_ptr // TODO should be Info only
              ) override;

  const DataSymmetriesForViewSegmentNumbers* get_symmetries_used() const override;

protected:
  //! variable that determines if a cylindrical FOV or the whole image will be handled
  bool restrict_to_cylindrical_FOV;

private:
  void actual_forward_project(RelatedViewgrams<float>&,
                              const DiscretisedDensity<3, float>&,
                              const int min_axial_pos_num,
                              const int max_axial_pos_num,
                              const int min_tangential_pos_num,
                              const int max_tangential_pos_num) override;
#if 0 // disabled as currently not used. needs to be written in the new style anyway
  void actual_forward_project(Bin&,
                              const DiscretisedDensity<3,float>&);
#endif

  // KT 20/06/2001 changed type from 'const DataSymmetriesForViewSegmentNumbers *'
  shared_ptr<DataSymmetriesForBins_PET_CartesianGrid> symmetries_ptr;
  /*
    The version which uses all possible symmetries.
    Here 0<=view < num_views/4 (= 45 degrees)
    */

  void forward_project_all_symmetries(Viewgram<float>& pos_view,
                                      Viewgram<float>& neg_view,
                                      Viewgram<float>& pos_plus90,
                                      Viewgram<float>& neg_plus90,
                                      Viewgram<float>& pos_min180,
                                      Viewgram<float>& neg_min180,
                                      Viewgram<float>& pos_min90,
                                      Viewgram<float>& neg_min90,
                                      const VoxelsOnCartesianGrid<float>& image,
                                      const int min_axial_pos_num,
                                      const int max_axial_pos_num,
                                      const int min_tangential_pos_num,
                                      const int max_tangential_pos_num) const;

  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void forward_project_view_plus_90_and_delta(Viewgram<float>& pos_view,
                                              Viewgram<float>& neg_view,
                                              Viewgram<float>& pos_plus90,
                                              Viewgram<float>& neg_plus90,
                                              const VoxelsOnCartesianGrid<float>& image,
                                              const int min_axial_pos_num,
                                              const int max_axial_pos_num,
                                              const int min_tangential_pos_num,
                                              const int max_tangential_pos_num) const;
  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void forward_project_view_min_180_and_delta(Viewgram<float>& pos_view,
                                              Viewgram<float>& neg_view,
                                              Viewgram<float>& pos_min180,
                                              Viewgram<float>& neg_min180,
                                              const VoxelsOnCartesianGrid<float>& image,
                                              const int min_axial_pos_num,
                                              const int max_axial_pos_num,
                                              const int min_tangential_pos_num,
                                              const int max_tangential_pos_num) const;

  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees
    (or others if the number of views is not a multiple of 4)
    Here 0<=view < num_views/2 (= 90 degrees)
    */
  void forward_project_delta(Viewgram<float>& pos_view,
                             Viewgram<float>& neg_view,
                             const VoxelsOnCartesianGrid<float>& image,
                             const int min_axial_pos_num,
                             const int max_axial_pos_num,
                             const int min_tangential_pos_num,
                             const int max_tangential_pos_num) const;

  //////////////// 2D
  void forward_project_all_symmetries_2D(Viewgram<float>& pos_view,
                                         Viewgram<float>& pos_plus90,
                                         Viewgram<float>& pos_min180,
                                         Viewgram<float>& pos_min90,
                                         const VoxelsOnCartesianGrid<float>& image,
                                         const int min_axial_pos_num,
                                         const int max_axial_pos_num,
                                         const int min_tangential_pos_num,
                                         const int max_tangential_pos_num) const;
  void forward_project_view_plus_90_2D(Viewgram<float>& pos_view,
                                       Viewgram<float>& pos_plus90,
                                       const VoxelsOnCartesianGrid<float>& image,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num,
                                       const int min_tangential_pos_num,
                                       const int max_tangential_pos_num) const;
  void forward_project_view_min_180_2D(Viewgram<float>& pos_view,
                                       Viewgram<float>& pos_min180,
                                       const VoxelsOnCartesianGrid<float>& image,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num,
                                       const int min_tangential_pos_num,
                                       const int max_tangential_pos_num) const;
  // no symmetries
  void forward_project_view_2D(Viewgram<float>& pos_view,
                               const VoxelsOnCartesianGrid<float>& image,
                               const int min_axial_pos_num,
                               const int max_axial_pos_num,
                               const int min_tangential_pos_num,
                               const int max_tangential_pos_num) const;
  //! The actual implementation of Siddon's algorithm
  /*! \return true if the LOR intersected the image, i.e. of Projptr (potentially) changed */
  template <int symmetry_type>
  static bool proj_Siddon(Array<4, float>& Projptr,
                          const VoxelsOnCartesianGrid<float>&,
                          const shared_ptr<const ProjDataInfoCylindrical> proj_data_info_sptr,
                          const float cphi,
                          const float sphi,
                          const float delta,
                          const float s_in_mm,
                          const float R,
                          const int min_ax_pos_num,
                          const int max_ax_pos_num,
                          const float offset,
                          const int num_planes_per_axial_pos,
                          const float axial_pos_to_z_offset,
                          const float norm_factor,
                          const bool restrict_to_cylindrical_FOV);

  void set_defaults() override;
  void initialise_keymap() override;
};
END_NAMESPACE_STIR
#endif
