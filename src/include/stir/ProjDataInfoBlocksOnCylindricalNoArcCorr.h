
/*
    Copyright (C) 2000- 2011-06-24, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2017, ETH Zurich, Institute of Particle Physics and Astrophysics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataInfoBlocksOnCylindricalNoArcCorr

  \author Kris Thielemans
  \author Parisa Khateri

*/
#ifndef __stir_ProjDataInfoBlocksOnCylindricalNoArcCorr_H__
#define __stir_ProjDataInfoBlocksOnCylindricalNoArcCorr_H__


#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindrical.h"
#include "stir/GeometryBlocksOnCylindrical.h"
#include "stir/DetectionPositionPair.h"
#include "stir/VectorWithOffset.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class Succeeded;
/*!
  \ingroup projdata
  \brief Projection data info for data which are not arc-corrected.

  For this class, 'tangential_pos_num' actually indexes an angular coordinate
  with a particular angular sampling (usually given by half the angle between
  detectors). That is
  \code
    get_s(Bin(..., tang_pos_num)) == ring_radius * sin(tang_pos_num*angular_increment)
  \endcode

  This class also contains some functions specific for (static) full-ring PET
  scanners. In this case, it is assumed that for 'raw' data (i.e. no mashing)
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


  \par Interchanging the 2 detectors

  When the ring difference = 0 (i.e. a 2D - or direct - sinogram),
  interchanging the 2 detectors does not change the LOR. This is why
  (in 2D) one gets away with a full sinogram size of
  num_views * 2 * num_views, where the size of 'detector-space' is
  twice as large.
  However, in 3D, interchanging the detectors, also interchanges the
  rings. One has 2 options:
  - have 1 sinogram with twice as many views, together with the rings
    as 'unordered pair' (i.e. ring_difference is always >0)
  - have 2 sinograms of the same size as in 2D, together with the rings
    as 'ordered pair' (i.e. ring_difference can be positive and negative).
  In STIR, we use the second convention.

  \todo The detector specific functions possibly do not belong in this class.
  One can easily imagine a case where the theta,phi,s,t coordinates are as
  described, but there is no real correspondence with detectors (for instance,
  a rotating system). Maybe they should be moved somewhere else?
  */
class ProjDataInfoBlocksOnCylindricalNoArcCorr : public ProjDataInfoGenericNoArcCorr
{
private:
  typedef ProjDataInfoGenericNoArcCorr base_type;
#ifdef SWIG
  // SWIG needs this typedef to be public
 public:
#endif
  typedef ProjDataInfoBlocksOnCylindricalNoArcCorr self_type;

public:
  //! Default constructor (leaves object in ill-defined state)
  ProjDataInfoBlocksOnCylindricalNoArcCorr();
  //! Constructor completely specifying all parameters
  /*! \see ProjDataInfoCylindrical class documentation for info on parameters */
  ProjDataInfoBlocksOnCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
    const float ring_radius, const float angular_increment,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v,
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  //! Constructor which gets \a ring_radius and \a angular_increment from the scanner
  /*! \a angular_increment is determined as Pi divided by the number of detectors in a ring.
  \todo only suitable for full-ring PET scanners*/
   ProjDataInfoBlocksOnCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
    const  VectorWithOffset<int>& num_axial_pos_per_segment,
    const  VectorWithOffset<int>& min_ring_diff_v,
    const  VectorWithOffset<int>& max_ring_diff_v,
    const int num_views,const int num_tangential_poss);

  ProjDataInfo* clone() const;

  bool operator==(const self_type&) const;

  virtual std::string parameter_info() const;

  //! \name set of obsolete functions to go between bins<->LORs (will disappear!)
  //@{
  Succeeded find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
					             const CartesianCoordinate3D<float>& c1,
						     const CartesianCoordinate3D<float>& c2) const;

  void find_bin_given_cartesian_coordinates_of_detection(Bin& bin,
						  const CartesianCoordinate3D<float>& coord_1,
						  const CartesianCoordinate3D<float>& coord_2) const;
  //@}

private:

  virtual bool blindly_equals(const root_type * const) const;

};

END_NAMESPACE_STIR

#endif
