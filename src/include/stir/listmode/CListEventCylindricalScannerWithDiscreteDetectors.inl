//
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::CListEventCylindricalScannerWithDiscreteDetectors

  \author Kris Thielemans
  \author Nikos Efthimiou
  \author Elise Emond
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2017, 2022, University College London
    Copyright (C) 2017, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

CListEventCylindricalScannerWithDiscreteDetectors::
CListEventCylindricalScannerWithDiscreteDetectors(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
{
    this->uncompressed_proj_data_info_sptr = std::dynamic_pointer_cast< const ProjDataInfoCylindricalNoArcCorr >(proj_data_info_sptr->create_shared_clone());

    if (is_null_ptr(this->uncompressed_proj_data_info_sptr))
        error("CListEventCylindricalScannerWithDiscreteDetectors takes only ProjDataInfoCylindricalNoArcCorr. Abort.");
}

LORAs2Points<float>
CListEventCylindricalScannerWithDiscreteDetectors::
get_LOR() const
{
  LORAs2Points<float> lor;
  bool swap = this->get_delta_time() < 0.F;
  // provide somewhat shorter names for the 2 coordinates, taking swap into account
  CartesianCoordinate3D<float>& coord_1 = swap ? lor.p2() : lor.p1();
  CartesianCoordinate3D<float>& coord_2 = swap ? lor.p1() : lor.p2();

  DetectionPositionPair<> det_pos;
  this->get_detection_position(det_pos);
  assert(det_pos.pos1().radial_coord()==0);
  assert(det_pos.pos2().radial_coord()==0);

  // TODO we're using an obsolete function here which uses a different coordinate system
  this->get_uncompressed_proj_data_info_sptr()->
    find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2,
                                                         det_pos.pos1().axial_coord(),
                                                         det_pos.pos2().axial_coord(),
                                                         det_pos.pos1().tangential_coord(),
                                                         det_pos.pos2().tangential_coord(),
                                                         det_pos.timing_pos());
  // find shift in z
  const float shift = this->get_uncompressed_proj_data_info_sptr()->get_ring_spacing()*
    (this->get_uncompressed_proj_data_info_sptr()->get_scanner_ptr()->get_num_rings()-1)/2.F;
  coord_1.z() -= shift;
  coord_2.z() -= shift;

  return lor;
}

void
CListEventCylindricalScannerWithDiscreteDetectors::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  assert(dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(&proj_data_info) != 0);
  DetectionPositionPair<> det_pos;
  this->get_detection_position(det_pos);
  if (static_cast<ProjDataInfoCylindricalNoArcCorr const&>(proj_data_info).
      get_bin_for_det_pos_pair(bin, det_pos) == Succeeded::no)
    bin.set_bin_value(0);
  else
  {
    bin.set_bin_value(1);
  }
}

bool
CListEventCylindricalScannerWithDiscreteDetectors::
is_valid_template(const ProjDataInfo& proj_data_info) const
{
    if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(&proj_data_info)!= 0)
        return true;

    return false;
}

END_NAMESPACE_STIR
