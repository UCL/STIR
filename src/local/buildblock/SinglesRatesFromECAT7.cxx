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
using std::fstream;
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

float 
SinglesRatesFromECAT7:: get_singles_rate(const DetectionPosition<>& det_pos,float time) const
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

