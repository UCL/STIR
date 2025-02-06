
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
#include "stir/VectorWithOffset.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class Succeeded;
class ProjDataInfoTests;
class BlocksTests;

/*!
  \ingroup projdata
  \brief Projection data info for data from a scanner with discrete dtectors organised by blocks

  This class also contains 2 deprecated functions specific for (static) full-ring PET
  scanners. In this case, it is assumed that for 'raw' data (i.e. no mashing)
  sinogram space is 'interleaved'. See documentation for ProjDataInfoCylindricalNoArcCorr.

  \deprecated This class will be removed in v7.0.
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

  //! Constructor which gets geometry from the scanner
  ProjDataInfoBlocksOnCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_ptr,
                                           const VectorWithOffset<int>& num_axial_pos_per_segment,
                                           const VectorWithOffset<int>& min_ring_diff_v,
                                           const VectorWithOffset<int>& max_ring_diff_v,
                                           const int num_views,
                                           const int num_tangential_poss);

  ProjDataInfo* clone() const override;

  bool operator==(const self_type&) const;

  std::string parameter_info() const override;

private:
  //! \name set of obsolete functions to go between bins<->LORs (will disappear!)
  //@{
  Succeeded find_scanner_coordinates_given_cartesian_coordinates(int& det1,
                                                                 int& det2,
                                                                 int& ring1,
                                                                 int& ring2,
                                                                 const CartesianCoordinate3D<float>& c1,
                                                                 const CartesianCoordinate3D<float>& c2) const;

  void find_bin_given_cartesian_coordinates_of_detection(Bin& bin,
                                                         const CartesianCoordinate3D<float>& coord_1,
                                                         const CartesianCoordinate3D<float>& coord_2) const;
  //@}
  // give test classes access to the private members
  friend class ProjDataInfoTests;
  friend class BlocksTests;

private:
  bool blindly_equals(const root_type* const) const override;
};

END_NAMESPACE_STIR

#endif
