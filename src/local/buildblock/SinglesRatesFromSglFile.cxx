//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \brief Implementation of SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/

#include "stir/DetectionPosition.h"
#include "stir/IndexRange3D.h"
#include "local/stir/SinglesRatesFromSglFile.h"

#include <vector>
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
#include "stir/IO/stir_ecat7.h"
#endif
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
#endif



START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
const unsigned 
SinglesRatesFromSglFile::size_of_singles_record = 4*128;

const char * const 
SinglesRatesFromSglFile::registered_name = "Singles From Sgl File"; 

static inline 
unsigned long int
convert_4_bytes(unsigned char * buffer)
{
  if (ByteOrder::native == ByteOrder::big_endian)
    return buffer[0] + 256UL*(buffer[1] + 256UL*(buffer[2] + 256UL*buffer[3]));
  else
    return buffer[3] + 256UL*(buffer[2] + 256UL*(buffer[1] + 256UL*buffer[0]));
}


SinglesRatesFromSglFile::
SinglesRatesFromSglFile()
{}

Array<3,float> 
SinglesRatesFromSglFile::read_singles_from_sgl_file (const string& sgl_filename)
{
#ifndef HAVE_LLN_MATRIX
  error("Compiled without ECAT7 support\n");
#else
  ifstream singles_file(sgl_filename.c_str(), ios::binary);
  if (!singles_file)
  {
    error("\nSinglesRatesFromSglFile: Couldn't open \"%s\".\n", sgl_filename.c_str());
  }
    
  //first find out the size of the file
  singles_file.seekg(0, ios::end);
  const streampos end_stream_position = singles_file.tellg();
  if (!singles_file)
  {
    error("\nSinglesRatesFromSglFile: Couldn't seek to end of file %s.",sgl_filename.c_str());
  }

  // go to the beginning and read the singles header
  singles_file.seekg(0, ios::beg);
 
  if (!singles_file)
    error("\nSinglesRatesFromSglFile: Couldn't seek to start of file %s.",sgl_filename.c_str());
  {
    char buffer[sizeof(Main_header)];
    singles_file.read(buffer,sizeof(singles_main_header));
    if (!singles_file)
    {
      error("\nSinglesRatesFromSglFile: Couldn't read main_header from %s.",sgl_filename.c_str());
    }
    else
    {
      unmap_main_header(buffer, &singles_main_header);
      ecat::ecat7::find_scanner(scanner_sptr, singles_main_header);
    }
   }
  if (scanner_sptr->get_type() != Scanner::E966)
    warning("check SinglesRatesFromSglFile for non-966\n");

  trans_blocks_per_bucket =scanner_sptr->get_num_transaxial_blocks_per_bucket();
  angular_crystals_per_block =scanner_sptr->get_num_transaxial_crystals_per_block();
  axial_crystals_per_block =scanner_sptr->get_num_axial_crystals_per_block();
  
  //skip the first 512 bytes which are part of ECAT7 header
  const int number_of_elements = 
    static_cast<int>((end_stream_position-static_cast<streampos>(512))/size_of_singles_record);

  //TODO move to Scanner
  if (scanner_sptr->get_type() == Scanner::E966)
    num_axial_blocks_per_singles_unit = 2;
  else
    num_axial_blocks_per_singles_unit = 1;

  singles = Array<3,float>(IndexRange3D(0,number_of_elements-1,
					 0,scanner_sptr->get_num_axial_blocks()/num_axial_blocks_per_singles_unit-1,
					0,scanner_sptr->get_num_transaxial_buckets()-1)); 
  Array<3,float>::full_iterator array_iter  = singles.begin_all();
 
  int singles_record_num=0;
  singles_file.seekg(512,ios::beg);
  while (singles_file && singles_record_num<=number_of_elements)
  {
    sgl_str singles_str;
    {
      unsigned char buffer[size_of_singles_record];
      singles_file.read(reinterpret_cast<char *>(buffer),size_of_singles_record);
      if (!singles_file)
	{
	  if (!singles_file.eof())
	    warning("Error reading singles file record %d. Stopped reading from this point.", singles_record_num);
	  break;
	}
      singles_str.time = convert_4_bytes(buffer);
      singles_str.num_sgl = convert_4_bytes(buffer+4);
      for (unsigned int i=0; i<(size_of_singles_record-8)/4; ++i)
	  singles_str.sgl[i] = convert_4_bytes(buffer+8+4*i);
    }
    const int num_singles_units =
      scanner_sptr->get_num_transaxial_buckets() *
      (scanner_sptr->get_num_axial_blocks()/num_axial_blocks_per_singles_unit);

    if (singles_str.num_sgl != num_singles_units)
      error("Number of singles units should be %d, but is %d in singles file",
	    num_singles_units,  singles_str.num_sgl);
    
    for ( int i = 0; i<num_singles_units;i++, ++array_iter)
    {
      assert(array_iter !=singles.end_all());
      *array_iter = static_cast<float>(singles_str.sgl[i]);
    }
    // singles in the sgl file given in msec.multiply with 0.001 to convert into sec.
    times.push_back(singles_str.time*0.001);
    ++singles_record_num;
  }
  
  assert(times.size()!=0);
  singles_time_interval = times[1] - times[0];
      
  if (singles_record_num!= number_of_elements)
  {
    error("\nSinglesRatesFromSglFile: Couldn't read all records in the .sgl file %s. Read %d of %d. Exiting\n",
	  sgl_filename.c_str(), singles_record_num, number_of_elements);
    //TODO resize singles to return array with new sizes
  }
#endif
   return singles;
    
}
  
vector<double> 
SinglesRatesFromSglFile::get_times() const
{
  return times;
}

float 
SinglesRatesFromSglFile::get_singles_rate(const DetectionPosition<>& det_pos,
					   const double start_time,  
					   const double end_time) const
{ 
  assert(end_time >= start_time);
  //cerr << start_time << "   ";
  // cerr << end_time << "   ";
  
  const int denom = trans_blocks_per_bucket*angular_crystals_per_block;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = axial_pos/(num_axial_blocks_per_singles_unit*axial_crystals_per_block);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;

  const float blocks_per_singles_unit =
    num_axial_blocks_per_singles_unit*trans_blocks_per_bucket;
  // SM this is pretty ugly but since sgl file has times from 2.008 all times less than this 
  // do not get assigned a value. In this case we take singles[0][ax][tang] for all times <2.008
  //TODO
  if ( start_time==end_time && start_time <=2.1)
    { 
      // TODO 4 966
     return  singles[0][axial_bucket_num][transaxial_bucket_num]/blocks_per_singles_unit;       
    }
    
#if 1
  // find out the index in the times vector (assumes almost constant sampling in time)

  /* The funny ranges below are made to reproduce Peter Bloomfield's code.

     For a given frame, the last singles to consider is found by
     taking the maximum index such that
       singles[last].time <  end_time_for_that_frame
     So, to find the first index, you have to find the last index in the previous frame,
     and then add 1.

     Instead of walking through the whole array to find the first index, we jump 
     to start_time/singles_time_interval (which is about 2 secs for the 966)
     which should get us pretty close. However, to be safe, we go a bit earlier,
     and then loop till we find the actual start.
  */
  //  const int min_index = singles.get_min_index();
  //const double singles_time_interval = singles[min_index+1].time - singles[min_index].time;
  const int min_index = 0;
  
  std::size_t start_index = 
    static_cast<std::size_t>(max(static_cast<int>(start_time/singles_time_interval) - 3, min_index));
  while (start_index<times.size() && times[start_index+1]<start_time)
  //while (start_index<times.get_max_index() && singles[start_index+1].time<start_time)
    ++start_index;

  float singles_average =0;
  std::size_t i;
  //for (i = start_index; i<=singles.get_max_index() && singles[i].time<end_time; i++)
  // SM correced upper range as times[i]<end_time is incorrect
  for (i = start_index; i<=times.size() && times[i]<end_time; i++)
    {
    singles_average += singles[i][axial_bucket_num][transaxial_bucket_num];       
    }
    return singles_average/(blocks_per_singles_unit*(i-start_index));
#else

    int i= (int)start_time/2;
    
    return  singles[i][axial_bucket_num][transaxial_bucket_num]/blocks_per_singles_unit;       


#endif
   

}


void 
SinglesRatesFromSglFile::
initialise_keymap()
{
  parser.add_start_key("Singles Rates From Sgl File");
  parser.add_key("sgl_filename", &sgl_filename);
  parser.add_stop_key("End Singles Rates From Sgl File");
}

bool 
SinglesRatesFromSglFile::
post_processing()
{
  read_singles_from_sgl_file(sgl_filename);
  return false;
}


void 
SinglesRatesFromSglFile::set_defaults()
{
  sgl_filename = "";
}


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR



