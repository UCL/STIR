/*
    Copyright (C) 2011, University College London
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataInfoGeneric
  
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Kris Thielemans
*/
#ifndef __stir_ProjDataInfoGeneric_H__
#define __stir_ProjDataInfoGeneric_H__


#include "stir/ProjDataInfoCylindrical.h"

START_NAMESPACE_STIR

class Succeeded;
template <typename coordT> class CartesianCoordinate3D;

/*!
  \ingroup projdata
  \brief projection data info for data corresponding to
  'Generic' sampling.

  This isn't completely generic as it assumes that there is "axial" (or "ring")
  coordinate and a "transaxial" (or "crystal"). However, their spacing can
  be arbitrary.

  Currently this class is derived from ProjDataInfoCylindrical. Arguably it
  should be the other way around. However, this would break backwards
  compatibility dramatically, and break other pull-requests in progress.
  We will leave this for later. At present, we just \c delete some member functions
  that do not make sense in the Generic case. There are a few ones left which might
  be removed in future, but are currently still used.

  \todo change hierarchy order, i.e. derive from ProjDataInfoCylindrical from ProjDataInfoGeneric.
*/

class ProjDataInfoGeneric: public ProjDataInfoCylindrical
{
private:
  typedef ProjDataInfo base_type;
  typedef ProjDataInfoGeneric self_type;

public:
  //! Type used by get_all_ring_pairs_for_segment_axial_pos_num()
  typedef std::vector<std::pair<int, int> > RingNumPairs;

  //! Constructors
  ProjDataInfoGeneric();
  //! Constructor given all the necessary information
  /*! The min and max ring difference in each segment are passed
  as VectorWithOffsets. All three vectors have to have index ranges
  from min_segment_num to max_segment_num.

  \warning Most of this library assumes that segment 0 corresponds
  to an average ring difference of 0.
  */
  ProjDataInfoGeneric(const shared_ptr<Scanner>& scanner_ptr,
    const VectorWithOffset<int>& num_axial_poss_per_segment, //index ranges from min_segment_num to max_segment_num
    const VectorWithOffset<int>& min_ring_diff,
    const VectorWithOffset<int>& max_ring_diff,
    const int num_views,const int num_tangential_poss);

  inline virtual float get_tantheta(const Bin&) const;

  inline float get_phi(const Bin&) const;

  inline float get_t(const Bin&) const;

  //! Return z-coordinate of the middle of the LOR
  /*!
  The 0 of the z-axis is chosen in the middle of the scanner.
  */
  inline float get_m(const Bin&) const;

  virtual void
    get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor, const Bin& bin) const;

  void set_azimuthal_angle_offset(const float angle) = delete;
  void set_azimuthal_angle_sampling(const float angle) = delete;

  //! set new number of views (currently calls error() unless nothing changes)
  virtual void
    set_num_views(const int new_num_views);

  float get_azimuthal_angle_sampling() const = delete;
  float get_azimuthal_angle_offset() const = delete;
  float get_ring_radius() const = delete;
  void set_ring_radii_for_all_views(const VectorWithOffset<float>& new_ring_radius) = delete;
  VectorWithOffset<float> get_ring_radii_for_all_views() const = delete;

  //! return an average ring-spacing (from Scanner)
  //TODOBLOCK what does this mean in a generic case?
  inline float get_ring_spacing() const;

  virtual inline float get_sampling_in_t(const Bin&) const;
  virtual inline float get_sampling_in_m(const Bin&) const;

  inline float get_axial_sampling(int segment_num) const override;
  inline bool axial_sampling_is_uniform() const override;

  virtual std::string parameter_info() const;

private:
  //! to be used in get LOR
  virtual void find_cartesian_coordinates_of_detection(CartesianCoordinate3D<float>& coord_1,
  													   CartesianCoordinate3D<float>& coord_2,
													   const Bin& bin) const = 0;
protected:
  CartesianCoordinate3D<float> z_shift;
};


END_NAMESPACE_STIR

#include "stir/ProjDataInfoGeneric.inl"

#endif
