/* CListRecordSAFIR.inl

 Coincidence Event Class for SAFIR: Inline File

	Copyright 2015 ETH Zurich, Institute of Particle Physics
	Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
	Copyright 2020 Positrigo AG, Zurich
    Copyright 2021 University College London

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

 */
/*!

  \file
  \ingroup listmode
  \brief Inline implementation of class stir::CListEventSAFIR and stir::CListRecordSAFIR with supporting classes

  \author Jannis Fischer
  \author Parisa Khateri
  \author Kris Thielemans
*/

#include<random>

#include "stir/LORCoordinates.h"
#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/Succeeded.h"

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

template <class Derived>
LORAs2Points<float>
CListEventSAFIR<Derived>::get_LOR() const
{
	LORAs2Points<float> lor;
	DetectionPositionPair<> det_pos_pair;

	static_cast<const Derived*>(this)->get_data().get_detection_position_pair(det_pos_pair);

    lor.p1() = map_to_use().get_coordinate_for_index(det_pos_pair.pos1());
    lor.p2() = map_to_use().get_coordinate_for_index(det_pos_pair.pos2());

    return lor;
}

namespace detail
{
template <class PDIT>
static inline bool
get_bin_for_det_pos_pair(Bin& bin, DetectionPositionPair<>& det_pos_pair, const ProjDataInfo& proj_data_info)
{
  if (auto proj_data_info_ptr = dynamic_cast<const PDIT*>(&proj_data_info))
    {
      if (proj_data_info_ptr->get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
        bin.set_bin_value(1);
      else
        bin.set_bin_value(-1);
      return true;
    }
  else
    return false;
}
} // namespace detail

template <class Derived>
void
CListEventSAFIR<Derived>::get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  DetectionPositionPair<> det_pos_pair;
  static_cast<const Derived*>(this)->get_data().get_detection_position_pair(det_pos_pair);

  if (!map_sptr)
    {
      // transform det_pos_pair into stir conventions
      det_pos_pair.pos1() = map_to_use().get_det_pos_for_index(det_pos_pair.pos1());
      det_pos_pair.pos2() = map_to_use().get_det_pos_for_index(det_pos_pair.pos2());
      
      if (det_pos_pair.pos1().tangential_coord() == det_pos_pair.pos2().tangential_coord()) {
        bin.set_bin_value(-1);
        return;
      }

      if (!detail::get_bin_for_det_pos_pair<ProjDataInfoGenericNoArcCorr>(bin, det_pos_pair, proj_data_info))
        {
          if (!detail::get_bin_for_det_pos_pair<ProjDataInfoCylindricalNoArcCorr>(bin, det_pos_pair, proj_data_info))
            error("Wrong type of proj-data-info for SAFIR");
        }
    }
  else
    {
      const stir::CartesianCoordinate3D<float> c1 = map_sptr->get_coordinate_for_index(det_pos_pair.pos1());
      const stir::CartesianCoordinate3D<float> c2 = map_sptr->get_coordinate_for_index(det_pos_pair.pos2());
      const LORAs2Points<float> lor(c1, c2);
      bin = proj_data_info.get_bin(lor);
    }
}

void CListEventDataSAFIR::get_detection_position_pair(DetectionPositionPair<>& det_pos_pair)
{
	det_pos_pair.pos1().radial_coord() = layerA;
	det_pos_pair.pos2().radial_coord() = layerB;

	det_pos_pair.pos1().axial_coord() = ringA;
	det_pos_pair.pos2().axial_coord() = ringB;

	det_pos_pair.pos1().tangential_coord() = detA;
	det_pos_pair.pos2().tangential_coord() = detB;
}

END_NAMESPACE_STIR
