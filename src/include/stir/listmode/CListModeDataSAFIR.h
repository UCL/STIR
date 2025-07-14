/* CListModeDataSAFIR.h

Coincidence LM Data Class for SAFIR: Header File
Jannis Fischer

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2020 Positrigo AG, Zurich

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
\brief Declaration of class stir::CListModeDataSAFIR

\author Jannis Fischer
*/

#ifndef __stir_listmode_CListModeDataSAFIR_H__
#define __stir_listmode_CListModeDataSAFIR_H__


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

#include "stir/listmode/CListRecordSAFIR.h"

START_NAMESPACE_STIR

/*!
  \brief Class for reading SAFIR listmode data with variable geometry
  \ingroup listmode
  \par
  By providing crystal map and template projection data files, the coordinates are read from files and used defining the LOR
  coordinates.
*/
template <class CListRecordT>
class CListModeDataSAFIR : public CListModeDataBasedOnCoordinateMap<CListRecordT>
{
public:

  CListModeDataSAFIR(const std::string& listmode_filename,
                     const std::string& crystal_map_filename,
                     const std::string& template_proj_data_filename,
                     const double lor_randomization_sigma = 0.0);

protected:
  virtual Succeeded open_lm_file() const;

};

END_NAMESPACE_STIR
#endif // CLISTMODEDATASAFIR_H
