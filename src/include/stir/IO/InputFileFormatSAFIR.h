/* InputFileFormatSAFIR.h

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

#ifndef __local_SAFIR_InputFileFormatSAFIR_H__
#define __local_SAFIR_InputFileFormatSAFIR_H__

#include <cstring>
#include <string>
#include <iostream>

#include "stir/IO/InputFileFormat.h"
#include "stir/utilities.h"
#include "stir/ParsingObject.h"

#include "stir/listmode/CListRecordSAFIR.h"
#include "stir/listmode/CListModeDataSAFIR.h"

START_NAMESPACE_STIR

//! Class for reading SAFIR coincidence listmode data.
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
		return this->actual_can_read(signature, input);
	}

	//! Checks in binary data file for correct signature (can be either "SAFIR CListModeData" or "MUPET CListModeData").
	virtual bool can_read( const FileSignature& signature, const std::string& filename) const
	{
		actual_do_parsing(filename);
		std::ifstream data_file(listmode_filename.c_str(), std::ios::binary);
		char* buffer = new char[32];
		data_file.read(buffer, 32);
		bool cr = !strncmp(buffer, "MUPET CListModeData\0", 20) ||  !strncmp(buffer, "SAFIR CListModeData\0", 20);
		delete[] buffer;
		return cr;
	}
	
protected:
	virtual bool actual_can_read(const FileSignature& signature, std::istream& input) const
	{
		actual_do_parsing(input);
		std::streampos pos = input.tellg();
		input.seekg(0, input.beg);
		char* buffer = new char[32];
		input.read(buffer, 32);
		input.seekg(pos);
		bool cr = !strncmp(buffer, "MUPET CListModeData\0", 20) ||  !strncmp(buffer, "SAFIR CListModeData\0", 20);
		delete[] buffer;
		return cr;
	}
	
public:

	virtual std::auto_ptr<data_type>
	read_from_file(std::istream& input) const
	{
		std::cerr << "InputFileFormatSAFIR: read_from_file(input_ifstream)" << std::endl;
		actual_do_parsing(input);
		return std::auto_ptr<data_type>(new CListModeDataSAFIR<CListRecordSAFIR>(listmode_filename, crystal_map_filename, template_proj_data_filename));
	}

	virtual std::auto_ptr<data_type> 
	read_from_file(const std::string& filename) const
	{
		std::cerr << "InputFileFormatSAFIR: read_from_file(" << filename << ")" << std::endl;
		actual_do_parsing(filename);
		return std::auto_ptr<data_type>(new CListModeDataSAFIR<CListRecordSAFIR>(listmode_filename, crystal_map_filename, template_proj_data_filename));
	}

private:
	typedef ParsingObject base_type;
	mutable std::string listmode_filename;
	mutable std::string crystal_map_filename;
	mutable std::string template_proj_data_filename;
	mutable bool did_parsing;

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
		template_proj_data_filename = "muppet.hs";
	}

	void actual_do_parsing(std::istream& input) const {
		if( did_parsing) return;
		// Ugly const_casts here, but I don't see an other nice way to use the parser
		const_cast<SAFIRCListmodeInputFileFormat*>(this)->parse(input);
		did_parsing = true;
		std::cerr << const_cast<SAFIRCListmodeInputFileFormat*>(this)->parameter_info();
	}

	void actual_do_parsing( const std::string& filename) const {
		if( did_parsing) return;
		// Ugly const_casts here, but I don't see an other nice way to use the parser
		const_cast<SAFIRCListmodeInputFileFormat*>(this)->parse(filename.c_str());
		did_parsing = true;
		std::cerr << const_cast<SAFIRCListmodeInputFileFormat*>(this)->parameter_info();
	}

};
END_NAMESPACE_STIR
#endif
