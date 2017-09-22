//
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::CListEventCylindricalScannerWithDiscreteDetectors

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

CListEventCylindricalScannerWithDiscreteDetectors::
CListEventCylindricalScannerWithDiscreteDetectors(const shared_ptr<ProjDataInfo>& proj_data_info)
{
  this->uncompressed_proj_data_info_sptr.reset
    (dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>
     (
     proj_data_info.get()));

    if (is_null_ptr(this->uncompressed_proj_data_info_sptr))
        error("CListEventCylindricalScannerWithDiscreteDetectors takes only ProjDataInfoCylindricalNoArcCorr. Abord.");
}

LORAs2Points<float>
CListEventCylindricalScannerWithDiscreteDetectors::
get_LOR() const
{
  LORAs2Points<float> lor;
  // provide somewhat shorter names for the 2 coordinates
  CartesianCoordinate3D<float>& coord_1 = lor.p1();
  CartesianCoordinate3D<float>& coord_2 = lor.p2();

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
                                                         det_pos.pos2().tangential_coord());
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
