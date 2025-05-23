/*
    Copyright (C) 2003-2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup singles_buildblock
  \brief Implementation of stir::ecat::ecat7::SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "stir/data/SinglesRatesFromSglFile.h"

#include <vector>
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
#include "stir/IO/stir_ecat7.h"
#endif
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
using std::ios;
#endif



START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
const unsigned 
SinglesRatesFromSglFile::SIZE_OF_SINGLES_RECORD = 4*128;

const char * const 
SinglesRatesFromSglFile::registered_name = "Singles From Sgl File"; 

static inline 
unsigned long int
convert_4_bytes(unsigned char * buffer)
{
  // The order from the file is always big endian. The native order doesn't matter
  // when converting by multiplying and adding the individual bytes.
  //if (ByteOrder::get_native_order() == ByteOrder::big_endian)
  //  return buffer[0] + 256UL*(buffer[1] + 256UL*(buffer[2] + 256UL*buffer[3]));
  //else
  return buffer[3] + 256UL*(buffer[2] + 256UL*(buffer[1] + 256UL*buffer[0]));

}



static inline
void
convert_int_to_4_bytes(unsigned long int val, unsigned char *buffer) {
  // Big endian
  buffer[0] = (val & 0xff000000) >> 24;
  buffer[1] = (val & 0x00ff0000) >> 16;
  buffer[2] = (val & 0x0000ff00) >> 8;
  buffer[3] = (val & 0x000000ff);
}





// Constructor
SinglesRatesFromSglFile::
SinglesRatesFromSglFile()
{}

int
SinglesRatesFromSglFile::
read_singles_from_sgl_file(const std::string& sgl_filename)
{

  int slice = 0;

#ifndef HAVE_LLN_MATRIX

  error("Compiled without ECAT7 support\n");

#else

  std::ifstream singles_file(sgl_filename.c_str(), std::ios::binary);
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't open \"%s\".\n", sgl_filename.c_str());
  }

  
  //first find out the size of the file
  singles_file.seekg(0, ios::end);
  const streampos end_stream_position = singles_file.tellg();
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't seek to end of file %s.",sgl_filename.c_str());
  }


  // go to the beginning and read the singles header
  singles_file.seekg(0, ios::beg);
 
  if (!singles_file) {
    error("\nSinglesRatesFromSglFile: Couldn't seek to start of file %s.",sgl_filename.c_str());
  }
  

  {
    char buffer[sizeof(Main_header)];
    singles_file.read(buffer,sizeof(_singles_main_header));
    if (!singles_file)
    {
      error("\nSinglesRatesFromSglFile: Couldn't read main_header from %s.",sgl_filename.c_str());
    }
    else
    {
      unmap_main_header(buffer, &_singles_main_header);
      ecat::ecat7::find_scanner(scanner_sptr, _singles_main_header);
    }
  }

  
  if (scanner_sptr->get_type() != Scanner::E966) {
    warning("check SinglesRatesFromSglFile for non-966\n");
  }


  // Get total number of bins for this type of scanner.
  const int total_singles_units = scanner_sptr->get_num_singles_units();

  // Calculate number of time slices from the length of the data (file size minus header).
  _num_time_slices =  
    static_cast<int>((end_stream_position - static_cast<streampos>(512)) /
                     SIZE_OF_SINGLES_RECORD);

   // Allocate the main array.
  _singles = Array<2, int>(IndexRange2D(0, _num_time_slices - 1, 0, total_singles_units - 1));

  
  singles_file.seekg(512, ios::beg);
  
  while (singles_file && slice < _num_time_slices) {
    
    // Temporary space to store file data.
    sgl_str singles_str;


    {
      unsigned char buffer[SIZE_OF_SINGLES_RECORD];
      
      singles_file.read(reinterpret_cast<char *>(buffer), SIZE_OF_SINGLES_RECORD);
      if (!singles_file) {
        
        if (!singles_file.eof()) {
          warning("Error reading singles file record %d. Stopped reading from this point.", 
                  slice);
        }

        break;
      }

      singles_str.time = convert_4_bytes(buffer);
      singles_str.num_sgl = convert_4_bytes(buffer+4);
      
      for (unsigned int i = 0; i < ( SIZE_OF_SINGLES_RECORD - 8)/4; ++i) {
        singles_str.sgl[i] = convert_4_bytes(buffer+8+4*i);
      }
    }

    
    if (singles_str.num_sgl != total_singles_units) {
      error("Number of singles units should be %d, but is %d in singles file",
	    total_singles_units,  singles_str.num_sgl);
    }
    


    // Copy the singles values to the main array.
    
    // Note. The singles values are arranged num_axial sets of num_transaxial
    // values.
    //
    // For a singles values for a unit at axial_index, transaxial_index
    // the values is found at single_str.sgl[]
    // singles_str.sgl[ transaxial_index + (axial_index * num_transaxial) ]
    //
    // The singles values are stored in the _singles array in the same order.
    // For other file formats the ordering of the units may be different.
    for (int singles_bin = 0; singles_bin < total_singles_units; ++singles_bin) {
      _singles[slice][singles_bin] = static_cast<int>(singles_str.sgl[singles_bin]);
    }
    
    
    // singles in the sgl file given in msec.multiply with 0.001 to convert into sec.
    _times.push_back(singles_str.time*0.001);

    // Add the last two words - total prompts and total randoms.
    _total_prompts.push_back(singles_str.sgl[total_singles_units]);
    _total_randoms.push_back(singles_str.sgl[total_singles_units + 1]);
    
    // Increment the slice index.
    ++slice;
    
  }
  
  assert(_times.size()!=0);
  _singles_time_interval = _times[1] - _times[0];
  
  if (slice != _num_time_slices)
  {
    error("\nSinglesRatesFromSglFile: Couldn't read all records in the .sgl file %s. Read %d of %d. Exiting\n",
	  sgl_filename.c_str(), slice, _num_time_slices);
    //TODO resize singles to return array with new sizes
  }

#endif

  // Return number of time slices read.
  return slice; 
    
}





// Write SinglesRatesFromSglFile to a singles file.
std::ostream& 
SinglesRatesFromSglFile::write(std::ostream& output) {

#ifndef HAVE_LLN_MATRIX

  error("Compiled without ECAT7 support\n");

#else
  
  char header_buffer[SIZE_OF_SINGLES_RECORD];
  unsigned char buffer[SIZE_OF_SINGLES_RECORD];
  
  memset(header_buffer, 0, SIZE_OF_SINGLES_RECORD);

  // Write header to buffer.
  map_main_header(header_buffer, &(this->_singles_main_header));
  
  // Write buffer to output.
  output.write(header_buffer, SIZE_OF_SINGLES_RECORD);

  if (!output) {
    error("\nSinglesRatesFromSglFile: Failed to write to output.");
    return(output);
  }
  
  
  // Empty buffer.
  memset(buffer, 0, SIZE_OF_SINGLES_RECORD);
  
  int total_singles_units = scanner_sptr->get_num_singles_units();
  unsigned long millisecs;
  
  // Write 512 byte blocks. One for each time slice recorded.
  for(int slice = 0 ; slice < _num_time_slices ; ++slice) {
    
    // Write data to buffer.
    millisecs = static_cast<unsigned long>(floor(_times[slice] * 1000.0));
 
    // Time and number of singles units
    convert_int_to_4_bytes(millisecs, buffer);
    convert_int_to_4_bytes(total_singles_units, buffer + 4);

    // Singles units
    // Note that the order of values in _singles is the same as that of the file.
    // This may not be the case for other file formats.
    for(int singles_bin = 0 ; singles_bin < total_singles_units ; ++singles_bin) {
      convert_int_to_4_bytes(_singles[slice][singles_bin], buffer + ((2 + singles_bin) * 4));
    }
    
    // Total prompts and total trues
    convert_int_to_4_bytes(_total_prompts[slice], buffer + ((2 + total_singles_units) * 4));
    convert_int_to_4_bytes(_total_randoms[slice], buffer + ((2 + total_singles_units + 1) * 4));
    
    
    // Write buffer to output.
    output.write(reinterpret_cast<char *>(buffer), SIZE_OF_SINGLES_RECORD);
    
    if (!output) {
      error("\nSinglesRatesFromSglFile: Failed to write to output.");
      break;
    }

  }
  

#endif

  return output;
}




/*
 *
 * Private methods.
 *
 */


void 
SinglesRatesFromSglFile::
initialise_keymap()
{
  parser.add_start_key("Singles Rates From Sgl File");
  parser.add_key("sgl_filename", &_sgl_filename);
  parser.add_stop_key("End Singles Rates From Sgl File");
}

bool 
SinglesRatesFromSglFile::
post_processing()
{
  read_singles_from_sgl_file(_sgl_filename);
  return false;
}


void 
SinglesRatesFromSglFile::set_defaults()
{
  _sgl_filename = "";
}








END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR



