/* CListRecordSAFIR.inl

 Coincidence Event Class for SAFIR: Inline File
 Jannis Fischer
 jannis.fischer@cern.ch

	Copyright 2015 ETH Zurich, Institute of Particle Physics

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

#include "stir/LORCoordinates.h"

#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfo.h"
#include "stir/Bin.h"
#include "stir/LORCoordinates.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <class Derived>
LORAs2Points<float>
CListEventSAFIR<Derived>::get_LOR() const
{
	LORAs2Points<float> lor;
	DetectionPositionPair<> det_pos_pair;

	static_cast<const Derived*>(this)->get_data().get_detection_position_pair(det_pos_pair);

	if(!map) stir::error("Crystal map not set.");

	lor.p1() = map->get_detector_coordinate(det_pos_pair.pos1());
	lor.p2() = map->get_detector_coordinate(det_pos_pair.pos2());

	return lor;
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
