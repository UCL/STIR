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
  scanner_sptr = find_scanner_from_ECAT_system_type(mptr->mhptr->system_type);

#if 0
  // TODO expand Scanner class such that these info can be obtaied from there
  EcatModel *ScannerModelInfo =  ecat_model(mptr->mhptr->system_type);
  transBlocksPerBucket=ScannerModelInfo->transBlocksPerBucket;
  angularCrystalsPerBlock=ScannerModelInfo->angularCrystalsPerBlock;
  axialCrystalsPerBlock =ScannerModelInfo->axialCrystalsPerBlock ;
#else
  trans_blocks_per_bucket =scanner_sptr->get_trans_blocks_per_bucket();
  angular_crystals_per_block =scanner_sptr->get_angular_crystals_per_block();
  axial_crystals_per_block =scanner_sptr->get_axial_crystals_per_block();
#endif
  Main_header* main_header = 
    reinterpret_cast<Main_header*>( mptr->mhptr ) ;

  // TODO find out sizes from somewhere somehow -- for HR++ 108 entries (3 (axial)*36(radial))
  singles =  Array<3,float>(IndexRange3D(1,main_header->num_frames,0,2,0,35)); 
  
  MatrixData* matrix ;
  vector<pair<double, double> > time_frames(main_header->num_frames);
  for ( int mat_frame = 1 ; mat_frame <= main_header->num_frames ; mat_frame++ )
  {
    //cerr << "Reading frame " << mat_frame <<endl;
    matrix= matrix_read( mptr, mat_numcod( mat_frame, 1, 1, 0, 0),
			 /*don't read the data*/MAT_SUB_HEADER) ;
    
    Scan3D_subheader* scan_subheader_ptr=  
      reinterpret_cast<Scan3D_subheader *>(matrix->shptr);
    time_frames[mat_frame-1].first=scan_subheader_ptr->frame_start_time/1000.;
    time_frames[mat_frame-1].second=
      time_frames[mat_frame-1].first +
      scan_subheader_ptr->frame_duration/1000.;

    float const* singles_ptr = reinterpret_cast<float const *>(scan_subheader_ptr->uncor_singles);//matrix->data_ptr);
    for(Array<2,float>::full_iterator iter = singles[mat_frame].begin_all(); iter != singles[mat_frame].end_all();)
    {
      *iter++ = *singles_ptr++;
    }
  }
  time_frame_defs =
    TimeFrameDefinitions(time_frames);
  return singles; 
  
}

float 
SinglesRatesFromECAT7:: get_singles_rate(const DetectionPosition<>& det_pos,
					 const double start_time,
					 const double end_time) const
{ 
  const int denom = trans_blocks_per_bucket*angular_crystals_per_block;
  const int axial_pos = det_pos.axial_coord();
  const int transaxial_pos = det_pos.tangential_coord();
  const int axial_bucket_num = axial_pos/(2*axial_crystals_per_block);//axialCrystalsPerBlock);
  const int transaxial_bucket_num = (transaxial_pos/denom) ;

  int frame_num = get_frame_number(start_time,end_time);
  //cerr << "Frame_num:   " << frame_num << endl;
  //cerr << " Axial pos: " << axial_bucket_num << endl;
  //cerr << " Transax pos: " << transaxial_bucket_num << endl;
  
  return singles[frame_num][axial_bucket_num][transaxial_bucket_num]/4.0;  // divide by 4.0 to be consistant with CTIs

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

int 
SinglesRatesFromECAT7::get_frame_number (const double start_time, const double end_time) const
{
  assert(end_time >=start_time);
  //cerr << "num frames are" << time_frame_defs.get_num_frames()<<endl;;
  for ( int i = 1; i <=time_frame_defs.get_num_frames(); i++)
    {
      double start = time_frame_defs.get_start_time(i);
      double end = time_frame_defs.get_end_time(i);
      //cerr << "Start frame" << start <<endl;
      //cerr << " End frame " << end << endl;
       if ((start/start_time)<=1 && (end/end_time)>=1)
	{
	  return i;
	}
    }
      
  error("SinglesRatesFromECAT7::get_frame_number didn't find a frame for time interval (%g,%g)\n",
	start_time, end_time);
  return 0; // to satisfy compilers
  

}


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


