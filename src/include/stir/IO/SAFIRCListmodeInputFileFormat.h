/* SAFIRCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for SAFIR.

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
  \brief Declaration of class stir::SAFIRCListmodeInputFileFormat

  \author Jannis Fischer
  \author Markus Jehl, Positrigo
*/

#ifndef __stir_IO_SAFIRCListmodeInputFileFormat_H__
#define __stir_IO_SAFIRCListmodeInputFileFormat_H__

#include <algorithm>
#include <cstring>
#include <string>
#include <iostream>

#include "boost/algorithm/string.hpp"

#include "stir/IO/InputFileFormat.h"
#include "stir/info.h"
#include "stir/error.h"
#include "stir/utilities.h"
#include "stir/ParsingObject.h"

#include "stir/listmode/CListRecordSAFIR.h"
#include "stir/listmode/CListModeDataSAFIR.h"

START_NAMESPACE_STIR

/*! \brief Class for reading SAFIR coincidence listmode data.

It reads a parameter file, which refers to
  - optional crystal map containing the mapping between detector index triple and cartesian coordinates of the crystal surfaces
(see DetectorCoordinateMap)
  - the binary data file with the coincidence listmode data in SAFIR format (see CListModeDataSAFIR)
  - a template projection data file, which defines the scanner

  If the map is not defined, the scanner detectors will be used. Otherwise, the nearest LOR of the scanner will be selected for
each event.

  An example of such a parameter file would be
  \code
        CListModeDataSAFIR Parameters:=
                listmode data filename:= listmode_input.clm.safir
                template projection data filename:= <projdata-filename>
        ; optional map specifying the actual location of the crystals
                crystal map filename:= crystal_map.txt
                ; optional random displacement of the LOR end-points in mm (only used of a map is present)
        LOR randomization (Gaussian) sigma:=0
        END CListModeDataSAFIR Parameters:=
  \endcode

  The first 32 bytes of the binary file are interpreted as file signature and matched against the strings "MUPET CListModeData\0",
"SAFIR CListModeData\0" and "NeuroLF CListModeData\0". If either is successfull, the class claims it can read the file format. The
rest of the file is read as records as specified as template parameter, e.g. CListRecordSAFIR.
*/
template <class EventDataType>
class SAFIRCListmodeInputFileFormat : public InputFileFormat<ListModeData>, public ParsingObject
{
public:
  SAFIRCListmodeInputFileFormat() {}
  const std::string get_name() const override { return "SAFIR Coincidence Listmode File Format"; }

  //! Checks in binary data file for correct signature.
  bool can_read(const FileSignature& signature, std::istream& input) const override
  {
    return false; // cannot read from istream
  }

  //! Checks in binary data file for correct signature (can be either "SAFIR CListModeData", "NeuroLF CListModeData" or "MUPET
  //! CListModeData").
  bool can_read(const FileSignature& signature, const std::string& filename) const override
  {
    // Looking for the right key in the parameter file
    std::ifstream par_file(filename.c_str());
    std::string key;
    std::getline(par_file, key, ':');
    key = standardise_interfile_keyword(key);
    if (key != std::string("clistmodedatasafir parameters"))
      {
        return false;
      }
    if (!actual_do_parsing(filename))
      return false;
    std::ifstream data_file(listmode_filename.c_str(), std::ios::binary);
    char* buffer = new char[32];
    data_file.read(buffer, 32);
    bool cr = false;
    // depending on used template, check header of listmode file for correct format
    if (std::is_same<EventDataType, CListEventDataSAFIR>::value)
      {
        cr = (!strncmp(buffer, "MUPET CListModeData\0", 20) || !strncmp(buffer, "SAFIR CListModeData\0", 20));
      }
    else if (std::is_same<EventDataType, CListEventDataNeuroLF>::value)
      {
        cr = !strncmp(buffer, "NeuroLF CListModeData\0", 20);
      }
    else
      {
        warning("SAFIRCListModeInputFileFormat was initialised with an unexpected template.");
      }

    if (!cr)
      {
        warning("SAFIRCListModeInputFileFormat tried to read file " + listmode_filename
                + " but it seems to have the wrong signature.");
      }

    delete[] buffer;
    return cr;
  }

  std::unique_ptr<data_type> read_from_file(std::istream& input) const override
  {
    error("read_from_file for SAFIRCListmodeData with istream not implemented %s:%d. Sorry", __FILE__, __LINE__);
    return unique_ptr<data_type>();
  }

  std::unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    info("SAFIRCListmodeInputFileFormat: read_from_file(" + std::string(filename) + ")");
    actual_do_parsing(filename);
    return std::unique_ptr<data_type>(new CListModeDataSAFIR<CListRecordSAFIR<EventDataType>>(
        listmode_filename, crystal_map_filename, template_proj_data_filename, lor_randomization_sigma));
  }

protected:
  typedef ParsingObject base_type;
  mutable std::string listmode_filename;
  mutable std::string crystal_map_filename;
  mutable std::string template_proj_data_filename;
  mutable double lor_randomization_sigma;

  bool actual_can_read(const FileSignature& signature, std::istream& input) const override
  {
    return false; // cannot read from istream
  }

  void initialise_keymap() override
  {
    base_type::initialise_keymap();
    this->parser.add_start_key("CListModeDataSAFIR Parameters");
    this->parser.add_key("listmode data filename", &listmode_filename);
    this->parser.add_key("crystal map filename", &crystal_map_filename);
    this->parser.add_key("template projection data filename", &template_proj_data_filename);
    this->parser.add_key("LOR randomization (Gaussian) sigma", &lor_randomization_sigma);
    this->parser.add_stop_key("END CListModeDataSAFIR Parameters");
  }

  void set_defaults() override
  {
    base_type::set_defaults();
    crystal_map_filename = "";
    template_proj_data_filename = "";
    lor_randomization_sigma = 0.0;
  }

  bool actual_do_parsing(const std::string& filename) const
  {
    // Ugly const_casts here, but I don't see an other nice way to use the parser
    if (const_cast<SAFIRCListmodeInputFileFormat<EventDataType>*>(this)->parse(filename.c_str()))
      {
        info(const_cast<SAFIRCListmodeInputFileFormat<EventDataType>*>(this)->parameter_info());
        return true;
      }
    else
      return false;
  }

  bool post_processing() override
  {
    if (!file_exists(listmode_filename))
      return true;
    else if (!file_exists(template_proj_data_filename))
      return true;
    else
      {
        return false;
      }
    return true;
  }

private:
  bool file_exists(const std::string& filename)
  {
    std::ifstream infile(filename.c_str());
    return infile.good();
  }
};
END_NAMESPACE_STIR
#endif
