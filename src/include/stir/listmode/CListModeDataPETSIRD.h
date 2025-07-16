/* CListModeDataPETSIRD.h

Coincidence LM Data Class for PETSIRD: Header File
Jannis Fischer

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2020 Positrigo AG, Zurich
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
\brief Declaration of class stir::CListModeDataPETSIRD

\author Daniel Deidda
*/

#ifndef __stir_listmode_CListModeDataPETSIRD_H__
#define __stir_listmode_CListModeDataPETSIRD_H__


#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/InputStreamWithRecords.h"
#include "stir/shared_ptr.h"

// #include "../../PETSIRD/cpp/helpers/include/petsird_helpers.h"
// #include "petsird_helpers/create.h"
// #include "petsird_helpers/geometry.h"

// #include "stir/listmode/CListRecordPETSIRD.h"

START_NAMESPACE_STIR

/*!
  \brief Class for reading PETSIRD listmode data with variable geometry
  \ingroup listmode
  \par
  By providing crystal map and template projection data files, the coordinates are read from files and used defining the LOR
  coordinates.
*/

class CListModeDataPETSIRD : public CListModeDataBasedOnCoordinateMap
{
public:

  CListModeDataPETSIRD(const std::string& listmode_filename,
                     const std::string& crystal_map_filename,
                     const std::string& template_proj_data_filename,
                     const double lor_randomization_sigma = 0.0);

  shared_ptr<CListRecord> get_empty_record_sptr() const override {

    return nullptr;
  }

  Succeeded get_next_record(CListRecord& record_of_general_type) const override{

    return Succeeded::no;
  }

protected:
  virtual Succeeded open_lm_file() const override;

};

END_NAMESPACE_STIR
#endif // CLISTMODEDATAPETSIRD_H
