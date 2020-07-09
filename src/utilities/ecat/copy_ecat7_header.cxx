//
//
/*
    Copyright (C) 2002 - 2009, Hammersmith Imanet Ltd
    Copyright 2011 - 2012, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*! 
\file
\ingroup utilities
\ingroup ECAT
\brief Copy contents of an ECAT7 header from 1 file to another

\par Usage

To copy the main header:
\verbatim
  copy_ecat7_header [--copy]  output_ECAT7_name input_ECAT7_name
\endverbatim
without --copy, num_planes etc are preserved.

To copy a subheader (but keeping essential info)
\verbatim
  copy_ecat7_header output_ECAT7_name f,g,d,b input_ECAT7_name f,g,d,b
\endverbatim

\author Kris Thielemans
*/


#include "stir/Succeeded.h"
#include "stir/utilities.h"
#include "stir/IO/stir_ecat7.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <errno.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
using std::string;
using std::ostream;
#endif

USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7
#define STIR_DO_IT(x) out_sh.x =in_sh.x;

void copy_subheader(Image_subheader& out_sh, const Image_subheader& in_sh)
{
  /*
	short data_type;
	short num_dimensions;
	short x_dimension;
	short y_dimension;
	short z_dimension;
	short align_0;
	float z_offset;
	float x_offset;
	float y_offset;
	float scale_factor;
	short image_min;
	short image_max;
	float x_pixel_size;
	float y_pixel_size;
	float z_pixel_size;
	short align_1;
	short align_2;
	short align_3;
  */
  if (out_sh.recon_zoom==0)
    {
      warning("Copying recon_zoom from template as original zoom is 0. I didn't check the pixel sizes though");
      STIR_DO_IT(recon_zoom);
    }
    else
    {
      if (fabs(in_sh.recon_zoom - out_sh.recon_zoom)<.05)
	warning("recon_zoom field in template (%g) and output (%g) is different.\n"
		"Keeping original value for zoom (%g) (also keeping the original voxel sizes)",
		out_sh.recon_zoom, in_sh.recon_zoom, in_sh.recon_zoom);
    }

  STIR_DO_IT(frame_duration);
  STIR_DO_IT(frame_start_time);
  STIR_DO_IT(filter_code);
  STIR_DO_IT(num_r_elements);
  STIR_DO_IT(num_angles);
  STIR_DO_IT(z_rotation_angle);
  STIR_DO_IT(decay_corr_fctr);
  STIR_DO_IT(processing_code);
  STIR_DO_IT(gate_duration);
  STIR_DO_IT(r_wave_offset);
  STIR_DO_IT(num_accepted_beats);
  STIR_DO_IT(filter_cutoff_frequency);
  STIR_DO_IT(filter_resolution);
  STIR_DO_IT(filter_ramp_slope);
  STIR_DO_IT(filter_order);
  STIR_DO_IT(filter_scatter_fraction);
  STIR_DO_IT(filter_scatter_slope);
  //STIR_DO_IT(x_resolution);
  //STIR_DO_IT(y_resolution);
  //STIR_DO_IT(z_resolution);
  for (int i=0; i<40; ++i)
    STIR_DO_IT(annotation[i]);
  STIR_DO_IT(mt_1_1);
  STIR_DO_IT(mt_1_2);
  STIR_DO_IT(mt_1_3);
  STIR_DO_IT(mt_2_1);
  STIR_DO_IT(mt_2_2);
  STIR_DO_IT(mt_2_3);
  STIR_DO_IT(mt_3_1);
  STIR_DO_IT(mt_3_2);
  STIR_DO_IT(mt_3_3);
  STIR_DO_IT(rfilter_cutoff);
  STIR_DO_IT(rfilter_resolution);
  STIR_DO_IT(rfilter_code);
  STIR_DO_IT(rfilter_order);
  STIR_DO_IT(zfilter_cutoff);
  STIR_DO_IT(zfilter_resolution);
  STIR_DO_IT(zfilter_code);
  STIR_DO_IT(zfilter_order);
  STIR_DO_IT(mt_1_4);
  STIR_DO_IT(mt_2_4);
  STIR_DO_IT(mt_3_4);
  STIR_DO_IT(scatter_type);
  STIR_DO_IT(recon_type);
  STIR_DO_IT(recon_views);
}


void copy_subheader(Image_subheader& out_sh, const Scan3D_subheader& in_sh)
{
  warning("Copying only timing and gating info from scan to image subheader\n");
  /*
	short data_type;
	short num_dimensions;
	short x_dimension;
	short y_dimension;
	short z_dimension;
	short align_0;
	float z_offset;
	float x_offset;
	float y_offset;
	float scale_factor;
	short image_min;
	short image_max;
	float x_pixel_size;
	float y_pixel_size;
	float z_pixel_size;
	short align_1;
	float x_resolution;
	float y_resolution;
	float z_resolution;
	short align_2;
	short align_3;

  //  if (out_sh.recon_zoom==0)
    {
      warning("Filling in recon_zoom as well\n");
      STIR_DO_IT(recon_zoom);
    }
#if 0
    else
    {
      warning("Keeping recon_zoom\n");
    }
#endif
  STIR_DO_IT(filter_code);
  STIR_DO_IT(z_rotation_angle);
  STIR_DO_IT(processing_code);
  STIR_DO_IT(filter_cutoff_frequency);
  STIR_DO_IT(filter_resolution);
  STIR_DO_IT(filter_ramp_slope);
  STIR_DO_IT(filter_order);
  STIR_DO_IT(filter_scatter_fraction);
  STIR_DO_IT(filter_scatter_slope);
  for (int i=0; i<40; ++i)
    STIR_DO_IT(annotation[i]);
  STIR_DO_IT(mt_1_1);
  STIR_DO_IT(mt_1_2);
  STIR_DO_IT(mt_1_3);
  STIR_DO_IT(mt_2_1);
  STIR_DO_IT(mt_2_2);
  STIR_DO_IT(mt_2_3);
  STIR_DO_IT(mt_3_1);
  STIR_DO_IT(mt_3_2);
  STIR_DO_IT(mt_3_3);
  STIR_DO_IT(rfilter_cutoff);
  STIR_DO_IT(rfilter_resolution);
  STIR_DO_IT(rfilter_code);
  STIR_DO_IT(rfilter_order);
  STIR_DO_IT(zfilter_cutoff);
  STIR_DO_IT(zfilter_resolution);
  STIR_DO_IT(zfilter_code);
  STIR_DO_IT(zfilter_order);
  STIR_DO_IT(mt_1_4);
  STIR_DO_IT(mt_2_4);
  STIR_DO_IT(mt_3_4);
  STIR_DO_IT(scatter_type);
  STIR_DO_IT(recon_type);
  STIR_DO_IT(recon_views);
  STIR_DO_IT(decay_corr_fctr);
*/
  STIR_DO_IT(frame_duration);
  STIR_DO_IT(frame_start_time);
  STIR_DO_IT(num_r_elements);
  STIR_DO_IT(num_angles);
  STIR_DO_IT(gate_duration);
  STIR_DO_IT(r_wave_offset);
  STIR_DO_IT(num_accepted_beats);
}

void copy_subheader(Scan3D_subheader& out_sh, const Scan3D_subheader& in_sh)
{
  /*
	short data_type;
	short num_dimensions;
	short num_r_elements;
	short num_angles;
	short num_z_elements[64];
	short ring_difference;
	short storage_order;
	short axial_compression;
	float x_resolution;
	float v_resolution;
	float z_resolution;
	float w_resolution;
        float scale_factor;
        short scan_min;	
        short scan_max;	
*/

  STIR_DO_IT(frame_start_time);
  STIR_DO_IT(frame_duration);
  STIR_DO_IT(prompts);
  STIR_DO_IT(net_trues);
  STIR_DO_IT(delayed);
  STIR_DO_IT(multiples);
  STIR_DO_IT(tot_avg_cor);
  STIR_DO_IT(tot_avg_uncor);
  STIR_DO_IT(loss_correction_fctr);
  STIR_DO_IT(corrections_applied);
  STIR_DO_IT(gate_duration);
  STIR_DO_IT(r_wave_offset);
  STIR_DO_IT(num_accepted_beats);
  STIR_DO_IT(total_coin_rate);
  for (int i=0; i<128; ++i)
    STIR_DO_IT(uncor_singles[i]);
}

void copy_subheader(Attn_subheader& out_sh, const Attn_subheader& in_sh)
{
  /*
	short data_type;
	short num_dimensions;
	short num_r_elements;
	short num_angles;
	short num_z_elements;
	short ring_difference;
	float scale_factor;
	short z_elements[64];
  */
  //STIR_DO_IT(x_resolution);
  //STIR_DO_IT(y_resolution);
  //STIR_DO_IT(z_resolution);
  //STIR_DO_IT(w_resolution);
  STIR_DO_IT(x_offset);
  STIR_DO_IT(y_offset);
  STIR_DO_IT(x_radius);
  STIR_DO_IT(y_radius);
  STIR_DO_IT(tilt_angle);
  STIR_DO_IT(attenuation_coeff);
  STIR_DO_IT(attenuation_type);
  STIR_DO_IT(attenuation_min);
  STIR_DO_IT(attenuation_max);
  STIR_DO_IT(skull_thickness);
  STIR_DO_IT(num_additional_atten_coeff);
  STIR_DO_IT(edge_finding_threshold);
  for (int i=0; i<8; ++i)
    STIR_DO_IT(additional_atten_coeff[i]);
}

void copy_subheader(MatrixData * data_out,
	            const MatrixData * data_in)
{
  MatrixFile *mptr = data_out->matfile;
  switch (mptr->mhptr->file_type)
    {
#if 0
    case CTISinogram:
	copy_subheader(
           *reinterpret_cast<Scan_subheader *>(data_out->shptr),
           *reinterpret_cast<Scan_subheader *>(data_in->shptr));
	break;
    case Normalization:
	copy_subheader(
           *reinterpret_cast<Norm_subheader *>(data_out->shptr),
           *reinterpret_cast<Norm_subheader *>(data_in->shptr));
	break;
#endif
    case PetImage:
    case ByteVolume:
    case PetVolume:
      {
	switch(data_in->matfile->mhptr->file_type)
	  {
	  case Byte3dSinogram:
	  case Short3dSinogram:
	  case Float3dSinogram :
	    copy_subheader(
			   *reinterpret_cast<Image_subheader *>(data_out->shptr),
			   *reinterpret_cast<Scan3D_subheader *>(data_in->shptr));
	    break;
	  case PetImage:
	  case ByteVolume:
	  case PetVolume:
	    copy_subheader(
			   *reinterpret_cast<Image_subheader *>(data_out->shptr),
			   *reinterpret_cast<Image_subheader *>(data_in->shptr));
	    break;
	  default:
	    error("\ncopy_subheader: cannot copy input subheader to subheader of type image\n");
	  }
	break;
      }
    case AttenCor:   		
      copy_subheader(
		     *reinterpret_cast<Attn_subheader *>(data_out->shptr),
		     *reinterpret_cast<Attn_subheader *>(data_in->shptr));
      break;
    case Byte3dSinogram:
    case Short3dSinogram:
    case Float3dSinogram :
      copy_subheader(
		     *reinterpret_cast<Scan3D_subheader *>(data_out->shptr),
		     *reinterpret_cast<Scan3D_subheader *>(data_in->shptr));
      break;
    default:
    case ByteProjection:
    case PetProjection:
      error("copy_subheader: file_type not supported yet\n");
    }
}

void update_main_header(Main_header& mh_out, const Main_header& mh_in)
  {
    Main_header mh = mh_in;
    mh.num_planes = mh_out.num_planes;
    mh.num_frames = mh_out.num_frames;
    mh.num_gates = mh_out.num_gates;
    mh.num_bed_pos = mh_out.num_bed_pos;
    mh.file_type = mh_out.file_type;
    
    mh_out = mh;
  }

Succeeded
copy_main_header(MatrixFile * mout_ptr, MatrixFile *min_ptr)
  {
    update_main_header(*mout_ptr->mhptr, *min_ptr->mhptr);
    
    if (mat_write_main_header(mout_ptr->fptr, mout_ptr->mhptr))
      return Succeeded::no;
    else
      return Succeeded::yes;
  }

class ECAT_dataset_spec
{
public:
ECAT_dataset_spec();
ECAT_dataset_spec(const char *const spec);
ECAT_dataset_spec(const string&);
int matnum() const;
int frame_num; 
int plane_num;
int gate_num; 
int data_num;
int bed_pos_num;
private:

void decode_spec(const char * const spec);
void set_defaults();
};

void
ECAT_dataset_spec::
set_defaults()
{
  frame_num=1;
  gate_num=1;
  data_num=0;
  bed_pos_num=0;
  plane_num=0;
}

void
ECAT_dataset_spec::
decode_spec(const char * const spec)
{
  set_defaults();
  sscanf(spec, "%d,%d,%d,%d",
                   &frame_num, &gate_num, &data_num, &bed_pos_num);
}

ECAT_dataset_spec::
ECAT_dataset_spec()
{
  set_defaults();
}

ECAT_dataset_spec::
ECAT_dataset_spec(const char * const spec)
{
  decode_spec(spec);
}

ECAT_dataset_spec::
ECAT_dataset_spec(const string& spec)
{
  decode_spec(spec.c_str());
}

int 
ECAT_dataset_spec::
matnum() const
{
  return mat_numcod (frame_num, 1, gate_num, data_num, bed_pos_num);
}

ostream& operator<<(ostream& s, const ECAT_dataset_spec& spec)
{
  s << spec.frame_num << ','
    << spec.gate_num << ','
    << spec.data_num << ','
    << spec.bed_pos_num;
  return s;
}
 
Succeeded
mat_write_any_subheader(
               MatrixData * data)
{
  struct MatDir matdir;
  
  matrix_errno = MAT_OK;
  matrix_errtxt[0] = '\0';
  if (data==NULL) 
    {
      matrix_errno = MAT_READ_FROM_NILFPTR;
      return Succeeded::no;
    }

  MatrixFile *mptr = data->matfile;
  if (mptr == NULL)
    { 
      matrix_errno = MAT_READ_FROM_NILFPTR ;
      return Succeeded::no;
    }
  else if (mptr->mhptr == NULL) matrix_errno = MAT_NOMHD_FILE_OBJECT ;
  else if (data->shptr == NULL) matrix_errno = MAT_NIL_SHPTR ;
  if (matrix_errno != MAT_OK) return Succeeded::no ;
 
  if (matrix_find (mptr, data->matnum, &matdir) != 0)
    return Succeeded::no;
  
  const int strtblk = matdir.strtblk;  

  int return_value=0;
  switch (mptr->mhptr->file_type)
    {
    case CTISinogram:
	return_value = mat_write_scan_subheader (mptr->fptr, mptr->mhptr, strtblk, 
           reinterpret_cast<Scan_subheader *>(data->shptr));
	break;
    case PetImage:
    case ByteVolume:
    case PetVolume:
	return_value = mat_write_image_subheader (mptr->fptr, mptr->mhptr, strtblk,
           reinterpret_cast<Image_subheader *>(data->shptr));
	break;
    case AttenCor:   		
	return_value = mat_write_attn_subheader (mptr->fptr, mptr->mhptr, strtblk,
           reinterpret_cast<Attn_subheader *>(data->shptr));
	break;
    case Normalization:
	return_value = mat_write_norm_subheader (mptr->fptr, mptr->mhptr, strtblk, 
           reinterpret_cast<Norm_subheader *>(data->shptr));
	break;
    case Byte3dSinogram:
    case Short3dSinogram:
    case Float3dSinogram :
	return_value = mat_write_Scan3D_subheader (mptr->fptr, mptr->mhptr, strtblk, 
           reinterpret_cast<Scan3D_subheader *>(data->shptr));
      
	break;
    default:
    case ByteProjection:
    case PetProjection:
    case PolarMap:
    case Norm3d:
	fprintf (stderr, "Not implemented yet\n");
	matrix_errno = MAT_WRITE_ERROR;
    }
    if (return_value)
    {
      matrix_perror("error in writing subheader");
      return Succeeded::no; 
    }
    else
      return Succeeded::yes;

}




int main(int argc, char *argv[])
{

  bool update=true; // switch between update and straight copy
  if (argc>1 && strcmp(argv[1], "--copy")==0)
    {
      update=false;
      --argc; ++ argv;
    }

  if(argc!=3 && argc!=5)
  {
    cerr<< "\nCopy contents of ECAT7 headers.\n"
        << "Usage: \n"
	<< "To copy the main header:\n"
	<< "\t" << argv[0] << " [--copy] output_ECAT7_name input_ECAT7_name\n"
        << "   without --copy, num_planes etc are preserved.\n"
	<< "or to copy a subheader (but keeping essential info)\n"
	<< "\t" << argv[0] << "  output_ECAT7_name f,g,d,b input_ECAT7_name f,g,d,b\n\n";
    return EXIT_FAILURE; 
  }

  const string output_name = argv[1];
  const string input_name = argc==3? argv[2] :argv[3];
 
  const bool write_main_header = argc==3; 

#ifndef USE_MATRIX_LIB_FOR_MAINHEADER
  if (write_main_header)
    {
      FILE * in_fptr = fopen(input_name.c_str(), "rb");
      if (!in_fptr) 
	{
	  error("Error opening '%s' for reading: %s", 
		input_name.c_str(), strerror(errno));
	}
      FILE * out_fptr = fopen(output_name.c_str(), "rb+");
      if (!out_fptr) 
	{
	  error("Error opening '%s' for reading and writing: %s", 
		output_name.c_str(), strerror(errno));
	}
      Main_header mh_in;
      if (mat_read_main_header(in_fptr, &mh_in)!=0)
	  error("Error reading main header from %s", input_name.c_str());
      if (update)
	{
	  Main_header mh_out;
	  if (mat_read_main_header(out_fptr, &mh_out)!=0)
	    error("Error reading main header from %s", output_name.c_str());
	  update_main_header(mh_out, mh_in);
	  if (mat_write_main_header(out_fptr, &mh_out))
	    error("Error writing main header to %s", output_name.c_str());
	}
      else
	{
	  if (mat_write_main_header(out_fptr, &mh_in))
	    error("Error writing main header to %s", output_name.c_str());
	}
      fclose(in_fptr);
      fclose(out_fptr);
      return EXIT_SUCCESS;
    }
#endif

  MatrixFile *min_ptr=
    matrix_open( input_name.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!min_ptr) {
    matrix_perror(input_name.c_str());
    exit(EXIT_FAILURE);
  }
  MatrixFile *mout_ptr=
    matrix_open( output_name.c_str(), MAT_OPEN_EXISTING, MAT_UNKNOWN_FTYPE);
  if (!mout_ptr) {
    matrix_perror(output_name.c_str());
    exit(EXIT_FAILURE);
  }
#ifdef USE_MATRIX_LIB_FOR_MAINHEADER
  if (write_main_header)
    {
      if (copy_main_header(mout_ptr, min_ptr) == Succeeded::no)
	return EXIT_FAILURE; 
      else
	return EXIT_SUCCESS;
    }
#endif

  assert(!write_main_header);

  const ECAT_dataset_spec out_spec(argv[2]);
  const ECAT_dataset_spec in_spec(argv[4]);
  cerr << "Attempting to read in '" << in_spec <<"' and out '"
       << out_spec << "'" << endl;
  MatrixData * mindata_ptr =
    matrix_read(min_ptr, in_spec.matnum(), MAT_SUB_HEADER);
  if (mindata_ptr == NULL)
  {
    matrix_perror("Error reading input subheader");
    return EXIT_FAILURE;
  }
  MatrixData * moutdata_ptr =
    matrix_read(mout_ptr, out_spec.matnum(), MAT_SUB_HEADER);
  if (moutdata_ptr == NULL)
  {
    matrix_perror("Error reading output subheader");
    return EXIT_FAILURE;
  }

  copy_subheader(moutdata_ptr, mindata_ptr);
  if (mat_write_any_subheader(moutdata_ptr) == Succeeded::no)
    return EXIT_FAILURE; 

  free_matrix_data(moutdata_ptr);
  free_matrix_data(mindata_ptr);
  matrix_close(mout_ptr);
  matrix_close(min_ptr);
  return EXIT_SUCCESS;
}
