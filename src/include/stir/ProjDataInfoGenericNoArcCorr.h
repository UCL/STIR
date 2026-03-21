/*
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    Copyright (C) 2026, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfoGenericNoArcCorr

  \author Kris Thielemans
  \author Parisa Khateri
  \author Michael Roethlisberger
*/
#ifndef __stir_ProjDataInfoGenericNoArcCorr_H__
#define __stir_ProjDataInfoGenericNoArcCorr_H__

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/DetectionPositionPair.h"
#include "stir/VectorWithOffset.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup projdata
  \brief Projection data info for data for a scanner with discrete detectors

  This isn't completely generic as it assumes that there is "axial" (or "ring")
  coordinate and a "transaxial" (or "crystal"). However, their spacing can
  be arbitrary.

  It is assumed that for 'raw' data (i.e. no mashing)
  sinogram space is 'interleaved': 2 adjacent LOR_angles are
  merged to 1 'view', while the corresponding bins are
  interleaved:

  \verbatim
     before interleaving               after interleaving
     a00     a01     a02 ...      view 0: a00 a10 a01 a11  ...
         a10     a11     ...
     a20     a21     a22 ...      view 1: a20 a30 a21 a31 ...
         a30     a31     ...
         \endverbatim
  This (standard) interleaving is done because for 'odd' LOR_angles there
  is no LOR which goes through the origin.

  \par Design considerations

  Currently this class is derived from ProjDataInfoCylindricalNoArcCorr. Arguably it
  should be the other way around. However, this would break backwards
  compatibility dramatically, and break other pull-requests in progress.
  We will leave this for later. At present, we just \c delete some member functions
  that do not make sense in the Generic case. There are a few ones left which might
  be removed in future, but are currently still used.

  \todo change ProjDataInfoCylindrical hierarchy order.

  */
class ProjDataInfoGenericNoArcCorr : public ProjDataInfoCylindricalNoArcCorr
{
private:
  typedef ProjDataInfoCylindricalNoArcCorr base_type;
#ifdef STIR_COMPILING_SWIG_WRAPPER
  // SWIG needs this typedef to be public
public:
#endif
  typedef ProjDataInfoGenericNoArcCorr self_type;

public:
  //! Type used by get_all_ring_pairs_for_segment_axial_pos_num()
  typedef std::vector<std::pair<int, int>> RingNumPairs;

  //! Default constructor (leaves object in ill-defined state)
  ProjDataInfoGenericNoArcCorr();

  //! Constructor which gets geometry from the scanner
  ProjDataInfoGenericNoArcCorr(
      const shared_ptr<Scanner> scanner_ptr,
      const VectorWithOffset<int>& num_axial_pos_per_segment, // index ranges from min_segment_num to max_segment_num
      const VectorWithOffset<int>& min_ring_diff_v,
      const VectorWithOffset<int>& max_ring_diff_v,
      const int num_views,
      const int num_tangential_poss);

  ProjDataInfo* clone() const override;

  bool operator==(const self_type&) const;

  //! Gets s coordinate in mm
  /*! \warning
    This does \c not take the 'interleaving' into account which is
    customarily applied to raw PET data.
  */
  inline float get_s(const Bin&) const override;

  inline float get_tantheta(const Bin&) const override;

  inline float get_phi(const Bin&) const override;

  inline float get_t(const Bin&) const override;

  //! Return z-coordinate of the middle of the LOR
  /*!
  The 0 of the z-axis is chosen in the middle of the scanner.
  */
  inline float get_m(const Bin&) const override;

  void get_LOR(LORInAxialAndNoArcCorrSinogramCoordinates<float>& lor, const Bin& bin) const override;

  void set_azimuthal_angle_offset(const float angle) = delete;
  void set_azimuthal_angle_sampling(const float angle) = delete;

  //! set new number of views (currently calls error() unless nothing changes)
  void set_num_views(const int new_num_views) override;

  float get_azimuthal_angle_sampling() const = delete;
  float get_azimuthal_angle_offset() const = delete;
  float get_ring_radius() const = delete;
  void set_ring_radii_for_all_views(const VectorWithOffset<float>& new_ring_radius) = delete;
  VectorWithOffset<float> get_ring_radii_for_all_views() const = delete;

  //! return an average ring-spacing (from Scanner)
  // TODOBLOCK what does this mean in a generic case?
  inline float get_ring_spacing() const;

  inline float get_sampling_in_t(const Bin&) const override;
  inline float get_sampling_in_m(const Bin&) const override;

  inline float get_axial_sampling(int segment_num) const override;
  inline bool axial_sampling_is_uniform() const override;

  std::string parameter_info() const override;

  Bin get_bin(const LOR<float>&, const double delta_time = 0.0) const override;

  //! \name set of obsolete functions to go between bins<->LORs (will disappear!)
  //@{
  /*! \warning These function take a different convention for the axial coordinate
    compare to the get_m(), get_LOR() etc. In the current function, the axial coordinate (z)
    is zero in the first ring, while for get_m() etc it is zero in the centre of the scanner.

    \warning \a default timing_pos_num=0 is for backwards compatibility

    \obsolete
  */

  // This version uses the coordinate map
  virtual void find_cartesian_coordinates_given_scanner_coordinates(CartesianCoordinate3D<float>& coord_1,
                                                                    CartesianCoordinate3D<float>& coord_2,
                                                                    const int Ring_A,
                                                                    const int Ring_B,
                                                                    const int det1,
                                                                    const int det2,
                                                                    const int timing_pos_num = 0) const override;

  //@}
protected:
  CartesianCoordinate3D<float> z_shift;

protected:
  bool blindly_equals(const root_type* const) const override;
};

//! For backwards compatibility
using ProjDataInfoGeneric = ProjDataInfoGenericNoArcCorr;

END_NAMESPACE_STIR

#include "stir/ProjDataInfoGenericNoArcCorr.inl"

#endif
