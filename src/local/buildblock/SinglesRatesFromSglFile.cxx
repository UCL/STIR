//
// $Id: 
//
/*!

  \file
  \brief Implementation of SinglesRatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/SinglesRatesFromSglFile.h"
#include "stir/DetectionPosition.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/IndexRange3D.h"

#include "ecat_model.h"

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

#if 0
SinglesRatesFromSglFile::
SinglesRatesFromSglFile()
{}

Array<3,float> 
SinglesRatesFromSglFile::read_singles_from_sgl_file (const string& sgl_filname)
{
#if 0
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
  singles_file.seekg(0, ios::beg);
  int number_of_elements = (int)end_stream_position/512;

  singles = Array<3,long int>(IndexRange3D(0,number_of_elements-1,0,2,0,35)); 
  //times = Array<1,long int>(0,number_of_elements-1);
 // skip the first 512 bytes which are part of ECAT7 header
  singles_file.seekg(512,ios::beg);

  Array<3,long int>::full_iterator array_iter  = singles.begin_all();
  //Array<1,long int>::full_iterator times_iter = times.begin_all();
  
  while (!singles_file.eof())
  {
    singles_file.read((char*)&singles_str,sizeof(singles_str));
    //*times_iter = singles_str.time;
    //times_iter++;
   /* for ( int i = 1; i<=126;i++, ++array_iter)
    {
      *array_iter = singles_str.sgl[i];
      if (ByteOrder::native_order != ByteOrder::big_endian)
	ByteOrder::swap_order(*array_iter);
    }*/
  }

     return singles;
     #endif
}
  

float 
SinglesRatesFromSglFile:: get_singles_rate(const DetectionPosition<>& det_pos,
					   const float time,  
					   const float end_time) const
{ 
  const int denom = transBlocksPerBucket*angularCrystalsPerBlock;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = axial_pos/(2*axialCrystalsPerBlock);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;

  return singles[0][axial_bucket_num][transaxial_bucket_num]/4.0;  // divide by 4.0 to be consistant with CTIs

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

#endif
END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#if 0
  while (!singles_file.eof())
  {
    singles_file.read((char*)&singles_str,sizeof(singles_str));     
    vector_of_records.push_back(singles_str);
  }

  std::vector<sgl_str>::const_iterator singles_iter= vector_of_records.begin();
  
  Array<3,float>::full_iterator iter_array = singles.begin_all();
  for ( int frame = 1; frame <= num_frames; frame++)
  {
    double start_frame = time_def.get_start_time(frame);
    double end_frame =time_def.get_end_time(frame);
    int number_of_samples =0;
    long int sum_singles [126];

    while (!((*singles_iter).time !=end_frame))
    {
      for(int i =1; i <=126; i++)
	sum_singles[i] +=(*singles_iter).sgl[i];
      number_of_samples++;
      singles_iter++;  
    }

    for(int i =1; i <=126; i++)
	sum_singles[i] /=number_of_samples;
    
    for( int i=1; i<=126;i++)
    {
      *iter_array++ = sum_singles[i];
    }
  }
  return singles;
}
 
#endif
