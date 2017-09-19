/* SAFIRCListmodeInputFileFormat.h

 Class defining input file format for coincidence listmode data for SAFIR. 
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

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::SAFIRCListmodeInputFileFormat

  \author Jannis Fischer
*/

#ifndef __stir_IO_SAFIRCListmodeInputFileFormat_H__
#define __stir_IO_SAFIRCListmodeInputFileFormat_H__

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

/*! Class for reading SAFIR coincidence listmode data.

It reads a parameter file, which refers to 
  - crystal map containing the mapping between detector index triple and cartesian coordinates of the crystal surfaces (see DetectorCoordinateMapFromFile)
  - the binary data file with the coincidence listmode data in SAFIR format (see CListModeDataSAFIR)
  - a template projection data file, which is used to generate the virtual cylindrical scanner

  An example of such a parameter file would be
  \code
	CListModeDataSAFIR Parameters:=
		listmode data filename:= listmode_input.clm.safir
		; the following two examples are also default to the key parser
		crystal map filename:= crystal_map_front.txt 
		template projection data filename:= safir_20.hs
	END CListModeDataSAFIR Parameters:=
  \endcode

  The first 32 bytes of the binary file are interpreted as file signature and matched against the strings "MUPET CListModeData\0" and "SAFIR CListModeData\0". If either is successfull, the class claims it can read the file format. The rest of the file is read as records as specified as template parameter, e.g. CListRecordSAFIR.
*/
class SAFIRCListmodeInputFileFormat : public InputFileFormat<CListModeData>, public ParsingObject
{
public:
	SAFIRCListmodeInputFileFormat() : did_parsing(false) {}
	virtual const std::string get_name() const
	{
		return "SAFIR Coincidence Listmode File Format";
	}
	
	//! Checks in binary data file for correct signature.
	virtual bool can_read(const FileSignature& signature, std::istream& input ) const
	{
		return false; // cannot read from istream
	}

	//! Checks in binary data file for correct signature (can be either "SAFIR CListModeData" or "MUPET CListModeData").
	virtual bool can_read( const FileSignature& signature, const std::string& filename) const
	{
		char* buffer = new char[20];
		
		std::ifstream par_file(filename.c_str(), std::ios::binary);
		par_file.read(buffer, 20);
		if( strncmp(buffer, "CListModeDataSAFIR", 18) ) { 
			delete[] buffer;
			return false;
		}

		bool can_parse = actual_do_parsing(filename);
		std::ifstream data_file(listmode_filename.c_str(), std::ios::binary);
		buffer = new char[32];
		data_file.read(buffer, 32);
		bool cr = (!strncmp(buffer, "MUPET CListModeData\0", 20) ||  !strncmp(buffer, "SAFIR CListModeData\0", 20)) && can_parse;
		
		delete[] buffer;
		return cr;
	}
	
	virtual std::unique_ptr<data_type>
	read_from_file(std::istream& input) const
	{
		error("read_from_file for SAFIRCListmodeData with istream not implemented %s:%d. Sorry",__FILE__, __LINE__);
		return unique_ptr<data_type>();
	}

	virtual std::unique_ptr<data_type> 
	read_from_file(const std::string& filename) const
	{
		info("SAFIRCListmodeInputFileFormat: read_from_file(" + std::string(filename) + ")");
		actual_do_parsing(filename);
		return std::unique_ptr<data_type>(new CListModeDataSAFIR<CListRecordSAFIR>(listmode_filename, crystal_map_filename, template_proj_data_filename));
	}

protected:
	typedef ParsingObject base_type;
	mutable std::string listmode_filename;
	mutable std::string crystal_map_filename;
	mutable std::string template_proj_data_filename;

	virtual bool actual_can_read(const FileSignature &signature, std::istream &input) const {
		return false; // cannot read from istream
	}

	void initialise_keymap() {
		base_type::initialise_keymap();
		this->parser.add_start_key("CListModeDataSAFIR Parameters");
		this->parser.add_key("listmode data filename", &listmode_filename);
		this->parser.add_key("crystal map filename", &crystal_map_filename);
		this->parser.add_key("template projection data filename", &template_proj_data_filename);
		this->parser.add_stop_key("END CListModeDataSAFIR Parameters");
	}

	void set_defaults() {
		base_type::set_defaults();
		crystal_map_filename = "crystal_map_front.txt";
		template_proj_data_filename = "safir_20.hs";
	}

	bool actual_do_parsing( const std::string& filename) const {
		if( did_parsing) return true;
		// Ugly const_casts here, but I don't see an other nice way to use the parser
		if( const_cast<SAFIRCListmodeInputFileFormat*>(this)->parse(filename.c_str()) ) {
			info(const_cast<SAFIRCListmodeInputFileFormat*>(this)->parameter_info());
			return true;
		}
		else return false;
	}

	bool post_processing() {
		if( !file_exists(listmode_filename) ) return true;
		else if( !file_exists(crystal_map_filename) ) return true; 
		else if( !file_exists(template_proj_data_filename) ) return true;
		else {
			did_parsing = true;
			return false;
		}
		return true;
	}
		


private:
	mutable bool did_parsing;
	bool file_exists( const std::string& filename) {
		std::ifstream infile(filename.c_str());
		return infile.good();
	}
	

};
END_NAMESPACE_STIR
#endif
