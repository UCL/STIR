//
//
/*!
  \file
  \ingroup listmode
  \brief Implementations of class stir::CListEventCylindricalScannerWithDiscreteDetectors
    
  \author Kris Thielemans
  \author Viet Dao
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2022, University College London
    Copyright (C) 2022, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/LORCoordinates.h"

START_NAMESPACE_STIR

CListEventCylindricalScannerWithDiscreteDetectors::
CListEventCylindricalScannerWithDiscreteDetectors(const shared_ptr<Scanner>& scanner_sptr)
  : scanner_sptr(scanner_sptr)
{
  this->proj_data_info_sptr.reset(
     ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
                                   1, scanner_sptr->get_num_rings()-1,
                                   scanner_sptr->get_num_detectors_per_ring()/2,
                                   scanner_sptr->get_default_num_arccorrected_bins(), 
                                   false));
  if(this->proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry()=="Cylindrical"){
    this->scanner_type = CYL;
  }else if(this->proj_data_info_sptr->get_scanner_ptr()->get_scanner_geometry()=="BlocksOnCylindrical"){
    this->scanner_type = BLOCK;
  }else{
    this->scanner_type = GEN;
  }
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
  switch (this->scanner_type)
  {
  case CYL:
    {
      this->get_uncompressed_proj_data_info_sptr()->
        find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2,
                                                             det_pos.pos1().axial_coord(),
                                                             det_pos.pos2().axial_coord(),
                                                             det_pos.pos1().tangential_coord(),
                                                             det_pos.pos2().tangential_coord());
      // find shift in z
      const float shift = this->scanner_sptr->get_ring_spacing()*
        (this->scanner_sptr->get_num_rings()-1)/2.F;
      coord_1.z() -= shift;
      coord_2.z() -= shift;
      break;
    }
  case BLOCK:
    {
      this->get_proj_data_info_sptr_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr>()->
        find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2,
                                                             det_pos.pos1().axial_coord(),
                                                             det_pos.pos2().axial_coord(),
                                                             det_pos.pos1().tangential_coord(),
                                                             det_pos.pos2().tangential_coord());
      const float shift = this->scanner_sptr->get_ring_spacing()*
        (this->scanner_sptr->get_num_rings()-1)/2.F;
      coord_1.z() -= shift;
      coord_2.z() -= shift;
      break;
    }
  case GEN:
    {
      this->get_proj_data_info_sptr_cast<ProjDataInfoGenericNoArcCorr>()->
        find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2,
                                                             det_pos.pos1().axial_coord(),
                                                             det_pos.pos2().axial_coord(),
                                                             det_pos.pos1().tangential_coord(),
                                                             det_pos.pos2().tangential_coord());
      const float shift = this->scanner_sptr->get_ring_spacing()*
        (this->scanner_sptr->get_num_rings()-1)/2.F;
      coord_1.z() -= shift;
      coord_2.z() -= shift;
      break;
    }
  }
      
  return lor;
}

void 
CListEventCylindricalScannerWithDiscreteDetectors::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  DetectionPositionPair<> det_pos;
  this->get_detection_position(det_pos);
  switch (this->scanner_type)
  {
  case CYL:
    {
      assert(dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(&proj_data_info) != 0);
      if (static_cast<ProjDataInfoCylindricalNoArcCorr const&>(proj_data_info).
          get_bin_for_det_pos_pair(bin, det_pos) == Succeeded::no)
        bin.set_bin_value(0);
      else
        bin.set_bin_value(1);
    }
    break;
  case BLOCK:
    {
      assert(dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr const*>(&proj_data_info)!= 0);
      if (static_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr const&>(proj_data_info).
          get_bin_for_det_pos_pair(bin, det_pos) == Succeeded::no)
        bin.set_bin_value(0);
      else
        bin.set_bin_value(1);
    }
    break;
  case GEN:
    {
      assert(dynamic_cast<ProjDataInfoGenericNoArcCorr const*>(&proj_data_info)!= 0);
      if (static_cast<ProjDataInfoGenericNoArcCorr const&>(proj_data_info).
          get_bin_for_det_pos_pair(bin, det_pos) == Succeeded::no)
        bin.set_bin_value(0);
      else
        bin.set_bin_value(1);
    }
    break;
  }
}

bool
CListEventCylindricalScannerWithDiscreteDetectors::
is_valid_template(const ProjDataInfo& proj_data_info) const
{
  if (*this->proj_data_info_sptr->get_scanner_sptr() != *proj_data_info.get_scanner_sptr())
    return false;
  switch (this->scanner_type)
  {
  case CYL:
    {
      return (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*>(&proj_data_info) != 0);
    }
  case BLOCK:
    {
      return (dynamic_cast<ProjDataInfoBlocksOnCylindricalNoArcCorr const*>(&proj_data_info)!= 0);
    }
  case GEN:
    {
      return (dynamic_cast<ProjDataInfoGenericNoArcCorr const*>(&proj_data_info)!= 0);
    }
  }
  return false;
}

END_NAMESPACE_STIR
