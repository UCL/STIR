/* CListRecordPETSIRD.inl

 Coincidence Event Class for PETSIRD: Inline File

        Copyright 2015 ETH Zurich, Institute of Particle Physics
        Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
        Copyright 2020, 2022 Positrigo AG, Zurich
        Copyright 2021 University College London
        Copyright 2025 National Physical Laboratory

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
  \brief Inline implementation of class stir::CListEventPETSIRD and stir::CListRecordPETSIRD with supporting classes

  \author Jannis Fischer
  \author Parisa Khateri
  \author Markus Jehl
  \author Kris Thielemans
  \author Daniel Deidda
*/

#include <random>

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
#include "stir/error.h"

START_NAMESPACE_STIR

LORAs2Points<float>
CListEventPETSIRD::get_LOR() const
{
  // LORAs2Points<float> lor;
  // DetectionPositionPair<> det_pos_pair;

  // // static_cast<const Derived*>(this)->get_data().get_detection_position_pair(det_pos_pair);

  // lor.p1() = map_to_use().get_coordinate_for_index(det_pos_pair.pos1());
  // lor.p2() = map_to_use().get_coordinate_for_index(det_pos_pair.pos2());

  // return lor;
}

void
CListEventPETSIRD::get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{}

END_NAMESPACE_STIR
