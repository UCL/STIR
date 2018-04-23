//
//

/*! 
\file
\ingroup utilities
\ingroup ECAT
\brief Show contents of ECAT7 header
\author Kris Thielemans
*/
/*
    Copyright (C) 2002- 2003, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/Succeeded.h"
#include "stir/utilities.h"
#include "stir/IO/stir_ecat7.h"

#include <iostream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
using std::string;
#endif

USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7

void dump_subheader(const Scan3D_subheader& scan3dsub)
{
  /*
	short data_type;
	short num_dimensions;
	short num_r_elements;
	short num_angles;
	short corrections_applied;
	short num_z_elements[64];
	short ring_difference;
	short storage_order;
	short axial_compression;
	float x_resolution;
	float v_resolution;
	float z_resolution;
	float w_resolution;
	unsigned int gate_duration;
	int r_wave_offset;
	int num_accepted_beats;
	float scale_factor;
	short scan_min;
	short scan_max;
	int prompts;
	int delayed;
	int multiples;
	int net_trues;
	float tot_avg_cor;
	float tot_avg_uncor;
	int total_coin_rate;
	unsigned int frame_start_time;
	unsigned int frame_duration;
	float 
*/
#define STIR_DUMPIT(x) \
  cout << "x = " << scan3dsub.x << '\n';

  cout << "frame_start_time = " << scan3dsub.frame_start_time << '\n';
  cout << "frame_duration = " << scan3dsub.frame_duration << '\n';
  STIR_DUMPIT(prompts);
  STIR_DUMPIT(net_trues);
  STIR_DUMPIT(delayed);
  STIR_DUMPIT(multiples);
  STIR_DUMPIT(tot_avg_uncor);
  cout << "loss_correction_fctr = " << scan3dsub.loss_correction_fctr << '\n';
  for (int i=0; i<128; ++i)
    cout << "uncor_singles[" << i << "] = " 
         << scan3dsub.uncor_singles[i] << '\n';
}


void
dump_subheader(MatrixFile * mptr,
	       const int frame_num, 
	       const int plane_num,
	       const int gate_num, 
	       const int data_num, 
	       const int bed_num)
{
  int matnum;
  struct MatDir matdir;
  Scan_subheader scansub;
  Image_subheader imagesub;
  Norm_subheader normsub;
  Attn_subheader attnsub;
  Scan3D_subheader scan3dsub;
  
  if (mptr->mhptr->sw_version < V7)
    matnum = mat_numcod (frame_num, plane_num, gate_num, data_num, bed_num);
  else
    matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
  
  if (matrix_find (mptr, matnum, &matdir) != 0)
    return;
  
  const int strtblk = matdir.strtblk;  
  switch (mptr->mhptr->file_type)
    {
#if 0
    case CTISinogram:
      {
	if (mat_read_scan_subheader (mptr->fptr, mptr->mhptr, strtblk, &scansub))
	  {
	    if (ferror(mptr->fptr))
	      perror("dump_subheader: error in reading subheader");
	    return;
	  }
	dump_subheader(scansub);
	break;
      }
    case PetImage:
    case ByteVolume:
    case PetVolume:
      {
	if (mat_read_image_subheader (mptr->fptr, mptr->mhptr, strtblk, &imagesub))
	  {
	    if (ferror(mptr->fptr))
	      perror("dump_subheader: error in reading subheader");
	    return;
	  }
      
	dump_subheader(imagesub);
	break;
      
      }
    
    case AttenCor:   		
      {
	if (mat_read_attn_subheader (mptr->fptr, mptr->mhptr, strtblk, &attnsub))
	  {
	    if (ferror(mptr->fptr))
	      perror("dump_subheader: error in reading subheader");
	    return;
	  }
	dump_subheader(attnsub);
	break;
      }
    
    
    case Normalization:
      {
	if (mat_read_norm_subheader (mptr->fptr, mptr->mhptr, strtblk, &normsub))
	  {
	    if (ferror(mptr->fptr))
	      perror("dump_subheader: error in reading subheader");
	    return;
	  }
	dump_subheader(normsub);
	break;
      }
#endif

    case Byte3dSinogram:
    case Short3dSinogram:
    case Float3dSinogram :
      {
	if (mat_read_Scan3D_subheader (mptr->fptr, mptr->mhptr, strtblk, &scan3dsub))
	  {
	    if (ferror(mptr->fptr))
	      perror("dump_subheader: error in reading subheader");
	    return;
	  }
      
	dump_subheader(scan3dsub);
	break;
      }
    default:
    case ByteProjection:
    case PetProjection:
    case PolarMap:
    case Norm3d:
      {
	fprintf (stderr, "Not implemented yet\n");
	break;
      }
    }

}




int main(int argc, char *argv[])
{
  std::string cti_name;
  
  if(argc==2)
  {
    cti_name=argv[1];
  }  
  else 
  {
    cerr<< "\nShow contents of ECAT7 headers.\n"
        << "Usage: \n"
	<< "\t" << argv[0] << "  ECAT7_name \n\n"
	<< "I will now ask you the same info interactively...\n\n";
    cti_name=ask_filename_with_extension("Name of the ECAT7 file? ", ".scn");
  }


  MatrixFile *mptr=
    matrix_open( cti_name.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror(cti_name.c_str());
    exit(EXIT_FAILURE);
  }
  
  const int num_frames = std::max(static_cast<int>( mptr->mhptr->num_frames),1);
  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_bed_poss = static_cast<int>( mptr->mhptr->num_bed_pos) + 1;
  const int num_gates = std::max(static_cast<int>( mptr->mhptr->num_gates),1);

  if (ask("Attempt all data-sets (Y) or single data-set (N)", true))
  {
    const int data_num=ask_num("Data number ? ",0,8, 0);

    for (int frame_num=1; frame_num<=num_frames;++frame_num)
      for (int bed_num=0; bed_num<num_bed_poss;++bed_num)
        for (int gate_num=1; gate_num<=num_gates;++gate_num)
          dump_subheader(mptr,
			 frame_num, 1, gate_num, data_num, bed_num);
  }
  else
  {
    const int frame_num=ask_num("Frame number ? ",1,num_frames, 1);
    const int bed_num=ask_num("Bed number ? ",0,num_bed_poss-1, 0);
    const int gate_num=ask_num("Gate number ? ",1,num_gates, 1);
    const int data_num=ask_num("Data number ? ",0,8, 0);
    
    dump_subheader(mptr,
		   frame_num, 1, gate_num, data_num, bed_num);
  }
  matrix_close(mptr);
  return EXIT_SUCCESS;
}
