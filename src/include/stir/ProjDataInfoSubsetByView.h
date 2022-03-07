/*
    Copyright (C) ...
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInfoSubsetByView

  \author Ashley Gillman
*/
#ifndef __stir_ProjDataInfoSubsetByView__H__
#define __stir_ProjDataInfoSubsetByView__H__


#include "stir/ProjDataInfo.h"
#include <utility>
#include <vector>

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup projdata
  \brief Projection data info for data corresponding to a subset sampling by views.

  The class maintains a reference to the 'original' fully sampled ProjData and defers to this
  object where possible.
*/
class ProjDataInfoSubsetByView: public ProjDataInfo
{
private:
  typedef ProjDataInfo base_type;
  typedef ProjDataInfoSubsetByView self_type;

public:
  //! Constructor setting relevant info for a ProjDataInfoSubsetByView
  /*!  org_proj_data_info_sptr is the original, fully sampled ProjDataInfo to subset.
       views are the views to subset over.
   */
  ProjDataInfoSubsetByView(const shared_ptr<const ProjDataInfo> org_proj_data_info_sptr,
                           const std::vector<int>& views);

  //! Clone the object.
  ProjDataInfoSubsetByView* clone() const override;

  //! true if the subset is actually all of the data
  bool contains_full_data() const;

  //! Get the view numbers of the original ProjDataInfo
  std::vector<int> get_original_view_nums() const;

  //! Get the Bin of the original ProjDataInfo corresponding to a Bin for this subset
  Bin get_original_bin(const Bin& bin) const;

  //! Get the Bin for this subset corresponding to a Bin of the original ProjDataInfo
  Bin get_bin_from_original(const Bin& org_bin) const;

  //! Set a new range of segment numbers for both this object and the original ProjDataInfo.
  /*!  \warning the new range has to be 'smaller' than the old one. */
  void reduce_segment_range(const int min_segment_num, const int max_segment_num) override;

  //! Invalid for a subset! This will call error()
  void
    set_num_views(const int new_num_views) override;

  //! Set number of tangential positions
  /*! \see ProjDataInfo::set_num_tangential_poss
   */
  void set_num_tangential_poss(const int num_tang_poss) override;

  //! Set number of axial positions per segment
  /*! \see ProjDataInfo::set_num_axial_poss_per_segment
    \param num_axial_poss_per_segment is a vector with the new numbers,
    where the index into the vector is the segment_num (i.e. it is not
    related to the storage order of the segments or so). */
  void set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment) override;

  //! Set minimum axial position number for 1 segment
  /*! \see ProjDataInfo::set_min_axial_pos_num
   */
  void set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num) override;

  //! Set maximum axial position number for 1 segment
  /*! \see ProjDataInfo::set_max_axial_pos_num
   */
  void set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num) override;

  //! Set minimum tangential position number
  /*! \see ProjDataInfo::set_min_tangential_pos_num
   */
  void set_min_tangential_pos_num(const int min_tang_poss) override;

  //! Set maximum tangential position number
  /*! \see ProjDataInfo::set_max_tangential_pos_num
   */
  void set_max_tangential_pos_num(const int max_tang_poss) override;

  //| \name Functions that return geometrical info for a Bin
  //@{

  //! Get tangent of the co-polar angle of the normal to the projection plane
  /*! theta=0 for 'direct' planes (i.e. projection planes parallel to the scanner axis)
      \see ProjDataInfo::get_tantheta
   */
  float get_tantheta(const Bin&) const override;

  //! Get azimuthal angle phi of the normal to the projection plane
  /*! phi=0 when the normal vector has no component along the horizontal axis
      \see ProjDataInfo::get_phi
   */
  float get_phi(const Bin&) const override;

  //! Get value of the (roughly) axial coordinate in the projection plane (in mm)
  /*! t-axis is defined to be orthogonal to the s-axis (and to the vector
      normal to the projection plane
      \see ProjDataInfo::get_t
   */
  float get_t(const Bin&) const override;

  //! Return z-coordinate of the middle of the LOR (in mm)
  /*!
    The middle is defined as follows: imagine a cylinder centred around
    the scanner axis. The LOR will intersect the cylinder at 2 opposite
    ends. The middle of the LOR is exactly halfway those 2 points.

    The 0 of the z-axis is chosen in the middle of the scanner.

    \see ProjDataInfo::get_m
  */
  float get_m(const Bin&) const override;

  //! Get value of the tangential coordinate in the projection plane (in mm)
  /*! s-axis is defined to be orthogonal to the scanner axis (and to the vector
      normal to the projection plane
      \see ProjDataInfo::get_s
   */
  float get_s(const Bin&) const override;

  //! Get LOR corresponding to a given bin
  /*!
      \see get_bin()
      \see ProjDataInfo::get_LOR
  */
  void
    get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>&,
	    const Bin&) const override;

  //@}

  //! \name Functions that return info on the sampling in the different coordinates
  //@{

  //! Get sampling distance in the \c t coordinate
  /*! For some coordinate systems, this might depend on the Bin. The
      default implementation computes it as
      \code
      1/2(get_t(..., ax_pos+1,...)-get_t(..., ax_pos-1,...)))
      \endcode

      \see ProjDataInfo::get_sampling_in_t
  */
  float get_sampling_in_t(const Bin&) const override;

  //! Get sampling distance in the \c m coordinate
  /*! For some coordinate systems, this might depend on the Bin. The
      default implementation computes it as
      \code
      1/2(get_m(..., ax_pos+1,...)-get_m(..., ax_pos-1,...)))
      \endcode

      \see ProjDataInfo::get_sampling_in_m
  */
  float get_sampling_in_m(const Bin&) const override;

  //! Get sampling distance in the \c s coordinate
  /*! For some coordinate systems, this might depend on the Bin. The
      default implementation computes it as
      \code
      1/2(get_s(..., tang_pos+1)-get_s(..., tang_pos_pos-1)))
      \endcode

      \see ProjDataInfo::get_sampling_in_s
  */
  float get_sampling_in_s(const Bin&) const override;

  //@}

  //! Find the bin in the projection data that 'contains' an LOR
  /*! Projection data corresponds to lines, so most Lines Of Response
      (LORs) there is a bin in the projection data. Usually this will be
      the bin which has a central LOR that is 'closest' to the LOR that
      is passed as an argument.

      If there is no such bin (e.g. the LOR does not intersect the
      detectors, Bin::get_bin_value() will be less than 0, otherwise
      it will be 1.

      \warning This function might get a different type of arguments
      in the next release.
      \see get_LOR()
      \see ProjDataInfo::get_bin
  */
  Bin get_bin(const LOR<float>&) const override;

  //! Check if \c *this contains \c proj
  /*!
     Like ProjDataInfo, will only compare for the same type with
     one expection: if this object contains the full data will
     attempt to compare the original ProjDataInfo against proj.
    */
  bool operator>=(const ProjDataInfo& proj) const override;

  std::string parameter_info() const override;

  //! Get a shared pointer to the original, fully sampled ProjDataInfo.
  shared_ptr<const ProjDataInfo> get_original_proj_data_info_sptr() const;

protected:
  bool blindly_equals(const root_type * const) const override;

private:

  shared_ptr<ProjDataInfo> org_proj_data_info_sptr;
  std::vector<int> view_to_org_view_num;
  std::vector<int> org_view_to_view_num;
};


END_NAMESPACE_STIR

//#include "stir/ProjDataInfoSubsetByView.inl"

#endif

