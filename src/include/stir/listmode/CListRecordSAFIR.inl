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

	if(!map) stir::error("Crystal map not set.");

	lor.p1() = map->get_detector_coordinate(det_pos_pair.pos1());
	lor.p2() = map->get_detector_coordinate(det_pos_pair.pos2());

	return lor;
}

//! author Parisa Khateri
//! Overrides the default implementation to use get_detection_position() which should be faster.
template <class Derived>
void
CListEventSAFIR<Derived>::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
	DetectionPositionPair<> det_pos_pair;
	static_cast<const Derived*>(this)->get_data().get_detection_position_pair(det_pos_pair);

	//check aligned detectors
  if (det_pos_pair.pos1().tangential_coord() == det_pos_pair.pos2().tangential_coord())
  {
		/*std::cerr<<"WARNING: aligned detectors: det1="<<det_pos_pair.pos1().tangential_coord()
              	<<"\tdet2="<<det_pos_pair.pos2().tangential_coord()
                <<"\tring1="<<det_pos_pair.pos1().axial_coord()
                <<"\tring2="<<det_pos_pair.pos2().axial_coord()<<"\n";*/
		bin.set_bin_value(-1);
  }
  // Case for generic scanner
  if(proj_data_info.get_scanner_ptr()->get_scanner_geometry() == "Generic" && bin.get_bin_value() != -1)
  {
      const ProjDataInfoGenericNoArcCorr& proj_data_info_gen =
                  dynamic_cast<const ProjDataInfoGenericNoArcCorr&>(proj_data_info);
      //transform det_pos_pair into stir coordinates
      DetectionPosition<> pos1 = det_pos_pair.pos1();
      DetectionPosition<> pos2 = det_pos_pair.pos2();
      det_pos_pair.pos1() = proj_data_info_gen.get_scanner_ptr()->get_detpos_from_id(pos1);
      det_pos_pair.pos2() = proj_data_info_gen.get_scanner_ptr()->get_detpos_from_id(pos2);

      if (proj_data_info_gen.get_bin_for_det_pos_pair(bin, det_pos_pair) == Succeeded::yes)
         bin.set_bin_value(1);
      else
         bin.set_bin_value(-1);
  }
  else
  {
	  if(!map) stir::error("Crystal map not set.");

      stir::CartesianCoordinate3D<float> c1 = map->get_detector_coordinate(det_pos_pair.pos1());
      stir::CartesianCoordinate3D<float> c2 = map->get_detector_coordinate(det_pos_pair.pos2());
      int det1, det2, ring1, ring2;

      if(proj_data_info.get_scanner_ptr()->get_scanner_geometry() == "Cylindrical"
                           && bin.get_bin_value()!=-1)
      {
         //const ProjDataInfoCylindricalNoArcCorr& proj_data_info_cyl =
          //         dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(proj_data_info);
                   
         LORAs2Points<float> lor;
       	 lor.p1() = c1;
         lor.p2() = c2;
         bin = proj_data_info.get_bin(lor);
      }
      else if(proj_data_info.get_scanner_ptr()->get_scanner_geometry() == "BlocksOnCylindrical"
                   				&& bin.get_bin_value()!=-1)
      {
    		const ProjDataInfoBlocksOnCylindricalNoArcCorr& proj_data_info_blk =
                    dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr&>(proj_data_info);

        if (proj_data_info_blk.find_scanner_coordinates_given_cartesian_coordinates(det1, det2, ring1, ring2, c1, c2) == Succeeded::no)
        		bin.set_bin_value(-1);
        else
        {
    			assert(!(ring1<0 ||
                   ring1>=proj_data_info_blk.get_scanner_ptr()->get_num_rings() ||
    							 ring2<0 ||
    							 ring2>=proj_data_info_blk.get_scanner_ptr()->get_num_rings())
    						 );
          
    			if(proj_data_info_blk.get_bin_for_det_pair(bin, det1, ring1, det2, ring2)==Succeeded::yes)
    					bin.set_bin_value(1);
          else
    					bin.set_bin_value(-1);
         }
      }
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
