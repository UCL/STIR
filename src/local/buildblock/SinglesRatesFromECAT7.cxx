//
// $Id$
//
/*!

  \file
  \brief Implementation of Singles RatesFromECAT7

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/SinglesRateFromECAT7.h"
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
SinglesRatesFromECAT7::registered_name = "Singles From ECAT7"; 

SinglesRatesFromECAT7::
SinglesRatesFromECAT7()
{}

Array<3,float> 
SinglesRatesFromECAT7::read_singles_from_file(const string& ECAT7_filename,
		   const ios::openmode open_mode)

{

  MatrixFile* mptr = matrix_open(ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);

  if (!(mptr->mhptr->file_type == Byte3dSinogram ||
    mptr->mhptr->file_type == Short3dSinogram || mptr->mhptr->file_type == Float3dSinogram))
    error("SinglesRatesFromECAT7: filename %s should be an ECAT7 emission file\n",
       ECAT7_filename.c_str());
  scanner_ptr = find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);
  // TODO expand Scanner class such that these info can be obtaied from there
  EcatModel *ScannerModelInfo =  ecat_model(mptr->mhptr->system_type);

  transBlocksPerBucket=ScannerModelInfo->transBlocksPerBucket;
  angularCrystalsPerBlock=ScannerModelInfo->angularCrystalsPerBlock;
  axialCrystalsPerBlock =ScannerModelInfo->axialCrystalsPerBlock ;

 
  Main_header* main_header = 
    reinterpret_cast<Main_header*>( mptr->mhptr ) ;

  // TODO find out sizes from somewhere somehow -- for HR++ 108 entries (3 (axial)*36(radial))
  singles =  Array<3,float>(IndexRange3D(0,main_header->num_frames-1,0,2,0,35)); 
  
  MatrixData* matrix ;
  for ( int mat_frame = 1 ; mat_frame <= main_header->num_frames ; mat_frame++ )
  {
    matrix= matrix_read( mptr, mat_numcod( mat_frame, 1, 1, 0, 0),/*don't read the data*/1) ;
    
    Scan3D_subheader* scan_subheader_ptr=  
      reinterpret_cast<Scan3D_subheader *>(matrix->shptr);
    
    float const* singles_ptr = reinterpret_cast<float const *>(scan_subheader_ptr->uncor_singles);//matrix->data_ptr);
    for(Array<3,float>::full_iterator iter = singles.begin_all(); iter != singles.end_all();)
    {
      *iter++ = *singles_ptr++;
    }
  }
  return singles; 
  
}
Array<3,float> 
SinglesRatesFromECAT7::read_singles_from_sgl_file (const string& singles_filename, 
						   TimeFrameDefinitions& time_def)
{

  ifstream singles_file(singles_filename.c_str(), ios::binary);
  if (!singles_file)
  {
    warning("\nCouldn't open %s.\n", singles_filename.c_str());
  }
  
  sgl_str singles_str;
  vector<sgl_str> vector_of_records;
  int num_frames = time_def.get_num_frames();

  singles = Array<3,float>(IndexRange3D(0,num_frames-1,0,2,0,35)); 
 
  // skip the first 512 bytes which are part of ECAT7 header
  singles_file.seekg(512,ios::beg);
  
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
 

float 
SinglesRatesFromECAT7:: get_singles_rate(const DetectionPosition<>& det_pos,
					 const float start_time,
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
SinglesRatesFromECAT7::
initialise_keymap()
{
  parser.add_start_key("Singles Rates From ECAT7");
  parser.add_key("ECAT7_filename", &ECAT7_filename);
  parser.add_stop_key("End Singles Rates From ECAT7");
}

bool 
SinglesRatesFromECAT7::
post_processing()
{
  read_singles_from_file(ECAT7_filename);
  return false;
}


void 
SinglesRatesFromECAT7::set_defaults()
{
  ECAT7_filename = "";
}

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#if 0
  //first find out the size of the file
  streampos current  = singles_file.tellg(); 
  singles_file.seekg(0, ios::end);
  streampos end_stream_position = singles_file.tellg();
  singles_file.seekg(0, ios::beg);
#endif

