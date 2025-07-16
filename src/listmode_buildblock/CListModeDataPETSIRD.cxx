/* CListModeDataPETSIRD.cxx

Coincidence LM Data Class for PETSIRD: Implementation

     Copyright 2015 ETH Zurich, Institute of Particle Physics
     Copyright 2020 Positrigo AG, Zurich
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
\brief implementation of class stir::CListModeDataPETSIRD

\author Jannis Fischer
\author Kris Thielemans
\author Markus Jehl
\author Daniel Deidda
*/
#include <iostream>
#include <fstream>

#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"

//#include "boost/static_assert.hpp"

#include "stir/listmode/CListModeDataPETSIRD.h"
// #include "stir/listmode/CListRecordPETSIRD.h"

using std::ios;
using std::fstream;
using std::ifstream;
using std::istream;

START_NAMESPACE_STIR;

CListModeDataPETSIRD::CListModeDataPETSIRD(const std::string& listmode_filename,
                                           const std::string& crystal_map_filename,
                                           const std::string& template_proj_data_filename,
                                           const double lor_randomization_sigma)
{
  CListModeDataBasedOnCoordinateMap::listmode_filename = listmode_filename;
  // if (!crystal_map_filename.empty())
  //   {
  //     this->map = MAKE_SHARED<DetectorCoordinateMap>(crystal_map_filename, lor_randomization_sigma);
  //   }
  // else
  //   {
  //     if (lor_randomization_sigma != 0)
  //       error("PETSIRD currently does not support LOR-randomisation unless a map is specified");
  //   }
  // shared_ptr<ExamInfo> _exam_info_sptr(new ExamInfo);
  // _exam_info_sptr->imaging_modality = ImagingModality::PT;
  // this->exam_info_sptr = _exam_info_sptr;

  //        // Here we are reading the scanner data from the template projdata
  // shared_ptr<ProjData> template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
  // this->set_proj_data_info_sptr(template_proj_data_sptr->get_proj_data_info_sptr()->create_shared_clone());

  // if (this->open_lm_file() == Succeeded::no)
  //   {
  //     error("CListModeDataPETSIRD: Could not open listmode file " + listmode_filename + "\n");
  //   }
}

Succeeded
CListModeDataPETSIRD::open_lm_file() const
{
  // shared_ptr<istream> stream_ptr(new fstream(this->listmode_filename.c_str(), ios::in | ios::binary));
  // if (!(*stream_ptr))
  //   {
  //     return Succeeded::no;
  //   }
  // info("CListModeDataPETSIRD: opening file \"" + this->listmode_filename + "\"", 2);
  // stream_ptr->seekg((std::streamoff)32);
  // this->current_lm_data_ptr.reset(
  //     new InputStreamWithRecords<CListRecordT, bool>(stream_ptr,
  //                                                    sizeof(CListTimeDataPETSIRD),
  //                                                    sizeof(CListTimeDataPETSIRD),
  //                                                    ByteOrder::little_endian != ByteOrder::get_native_order()));
  // return Succeeded::yes;
}

// template class CListModeDataPETSIRD<CListRecordPETSIRD>;

END_NAMESPACE_STIR
