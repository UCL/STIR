//
// $Id$
//
/*!

  \file
  \brief Implementation of SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir/DetectionPosition.h"
#include "stir/IndexRange3D.h"
#include "local/stir/SinglesRatesFromSglFile.h"

#include <vector>
#include "ecat_model.h"
#include "stir/IO/stir_ecat7.h"
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::streampos;
#endif



START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
const char * const 
SinglesRatesFromSglFile::registered_name = "Singles From Sgl File"; 


SinglesRatesFromSglFile::
SinglesRatesFromSglFile()
{}

Array<3,float> 
SinglesRatesFromSglFile::read_singles_from_sgl_file (const string& sgl_filname)
{
  ifstream singles_file(sgl_filname.c_str(), ios::binary);
  if (!singles_file)
  {
    warning("\nCouldn't open %s.\n", sgl_filname.c_str());
  }
  
  sgl_str singles_str;
  vector<sgl_str> vector_of_records;
  
  //first find out the size of the file
  streampos current  = singles_file.tellg(); 
  singles_file.seekg(0, ios::end);
  streampos end_stream_position = singles_file.tellg();

  // go to the beginning and read the singles header
  singles_file.seekg(0, ios::beg);
 
  if (!singles_file)
  {
    warning("\nCouldn't read main_header from %s.",sgl_filname.c_str());
  }
  else
  {
    char buffer[sizeof(Main_header)];
    singles_file.read(buffer,sizeof(singles_main_header));
    if (!singles_file)
    {
      warning("\nCouldn't read main_header from %s.",sgl_filname.c_str());
    }
    else
    {
      unmap_main_header(buffer, &singles_main_header);
      ecat::ecat7::find_scanner(scanner_sptr, singles_main_header);
    }
   }
  trans_blocks_per_bucket =scanner_sptr->get_trans_blocks_per_bucket();
  angular_crystals_per_block =scanner_sptr->get_angular_crystals_per_block();
  axial_crystals_per_block =scanner_sptr->get_axial_crystals_per_block();
  
  //skip the first 512 bytes which are part of ECAT7 header
  int number_of_elements = 
    static_cast<int>((end_stream_position-static_cast<streampos>(512))/sizeof(singles_str));

  singles = Array<3,float>(IndexRange3D(0,number_of_elements-1,0,2,0,35)); 
  Array<3,float>::full_iterator array_iter  = singles.begin_all();
 
  int singles_record_num=0;
  singles_file.seekg(512,ios::beg);
  while (singles_file && singles_record_num<=number_of_elements)
  {
    singles_file.read((char*)&singles_str,sizeof(singles_str));
    if (!singles_file)
      break;
    for ( int i = 0; i<=107;i++, ++array_iter)
    {
      assert(array_iter !=singles.end_all());
      if (ByteOrder::native != ByteOrder::big_endian)
	ByteOrder::swap_order(singles_str.sgl[i]);
      *array_iter = singles_str.sgl[i];      
    }
    if (ByteOrder::native != ByteOrder::big_endian)
	ByteOrder::swap_order(singles_str.time);
    // singles in the sgl file given in msec.multiply with 0.001 to convert into sec.
    times.push_back(singles_str.time*0.001);
    ++singles_record_num;
  }
  if (singles_record_num!= number_of_elements)
  {
    warning("\nCouldn't read all records in the .sgl file %s. Read %d of %d. Exiting\n",
	  sgl_filname.c_str(), singles_record_num, number_of_elements);
    //TODO resize singles to return array with new sizes
  }

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
#if 0
  static double start_previous = start_time;
  static double end_previous =  end_time;
#endif
  const int denom = trans_blocks_per_bucket*angular_crystals_per_block;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = axial_pos/(2*axial_crystals_per_block);//axialCrystalsPerBlock);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;

  // find out the index in the times vector (time samples are every 2sec, hence divide with 2)
  int start_index = (int)start_time/2;
  int end_index = (int)end_time/2;

  static float singles_average =0;
  
  for ( int i = start_index; i<=end_index; i++)
  {
    singles_average += singles[i][axial_bucket_num][transaxial_bucket_num];
  }
  return singles_average/4.0; //*count;  // divide by 4.0 to be consistant with CTIs

}
#if 0
float 
SinglesRatesFromSglFile::get_singles_rate(const DetectionPosition<>& det_pos,
					   const double start_time,  
					   const double end_time) const
{ 
  static int start_time_counter=0;
  static int beginning;
  static int ending;
  start_time_counter ++;
  static double tmp;
  if (start_time_counter ==1)
  {
    tmp = start_time;
  }

  const int denom = trans_blocks_per_bucket*angular_crystals_per_block;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = axial_pos/(2*axial_crystals_per_block);//axialCrystalsPerBlock);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;

  static int start_index;
  static int end_index;
  
  if (fabs(tmp-start_time) <.0000001 && start_time_counter==1)
  {
  for ( int i = 0; i<times.size(); i++)
  {
    double ttt= times[i];
    //sampling in the sgl file is ~ 2.0
    if (fabs(start_time-times[i]) < 2.05F)
    {
      start_index = i;
      beginning=i;
    }
    if (fabs(end_time-times[i]) < 2.05F)
    {
      end_index = i;
      ending =i;
    }
  }
  //start_time_counter =0;
  }
  

  static float singles_average =0;
  int count=0;
  if (beginning !=start_index && ending!=end_index)
  {
  for ( int i = start_index; i<=end_index; i++)
  {
    singles_average += singles[i][axial_bucket_num][transaxial_bucket_num];
    count ++;
  }
  }
    return singles_average/4.0*count;  // divide by 4.0 to be consistant with CTIs

}

#endif
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



