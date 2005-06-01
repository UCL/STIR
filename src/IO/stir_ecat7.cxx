//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
  \ingroup ECAT

  \brief Implementation of routines which convert ECAT7 things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author Cristina de Oliveira (offset_in_ECAT_file function)
  \warning This only works with some CTI file_types. In particular, it does NOT
  work with the ECAT6-like files_types, as then there are subheaders 'in' the 
  datasets.
  
  \warning Implementation uses the Louvain la Neuve Ecat library. So, it will
    only work on systems where this library works properly.

  $Date$
  $Revision$
*/

#ifdef HAVE_LLN_MATRIX

#include "stir/ProjDataInfo.h"
#include "stir/ProjDataFromStream.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/NumericInfo.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ByteOrder.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/SegmentByView.h"
#include "stir/IndexRange2D.h"
#include "stir/Scanner.h" 
#include "stir/Bin.h" 
#include "stir/Succeeded.h" 
#include "stir/convert_array.h"
#include "stir/NumericInfo.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/IO/write_data.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <memory>


#ifndef STIR_NO_NAMESPACES
using std::size_t;
using std::string;
using std::ios;
using std::iostream;
using std::fstream;
using std::cerr;
using std::endl;
using std::cout;
using std::copy;
#ifndef STIR_NO_AUTO_PTR
using std::auto_ptr;
#endif
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
/* ------------------------------------
 *	print_debug
 * ------------------------------------*/
static int print_debug (char const * const fname, char *format, ...) 
{
    va_list ap;
    char *fmt;
    int len;


    if (0)//flagged (fname) != NULL)
    {

	len = strlen (fname) + strlen (format) + 5;
	if ((fmt = (char *)calloc ((long)len, sizeof (char))) == NULL)
		return (1);
	sprintf (fmt, "%s%s%s", fname, " :: ", format);

    	va_start (ap, format);
	vfprintf (stderr, fmt, ap);

	free (fmt);
        va_end (ap);
    
    }

    return (0);

}


static bool is_ECAT7_file(Main_header& mhead, const string& filename)
{
  FILE * cti_fptr=fopen(filename.c_str(), "rb"); 
  // first check 'magic_number' before going into LLN routines 
  // to avoid crashes and error messages there
  // an ECAT7 file should start with MATRIX7
  {
    char magic[7];
    if (! cti_fptr ||
	fread(magic, 1, 7, cti_fptr) != 7 ||
	strncmp(magic, "MATRIX7",7)!=0)
      return false;
  }
  if(mat_read_main_header(cti_fptr, &mhead)!=0) 
    {
      // this is funny as it's just reading a bunch of bytes. anyway. we'll assume it isn't ECAT7
      return false;
    }
  else
    {
      // do some checks on the main header
      fclose(cti_fptr);
      return 
	mhead.sw_version>=70 && mhead.sw_version<=79  &&
	( mhead.file_type >= 1 && mhead.file_type <= Float3dSinogram) &&
	mhead.num_frames>0;
    }
}

bool is_ECAT7_file(const string& filename)
{
  Main_header mhead;
  return is_ECAT7_file(mhead, filename);
}

bool is_ECAT7_image_file(const string& filename)
{
  Main_header mhead;
  return is_ECAT7_file(mhead, filename) &&
    (mhead.file_type == PetImage ||
     mhead.file_type ==ByteVolume || mhead.file_type == PetVolume);
}


bool is_ECAT7_emission_file(const string& filename)
{
  Main_header mhead;
  return is_ECAT7_file(mhead, filename) &&
    (mhead.file_type == CTISinogram || mhead.file_type == Byte3dSinogram||
    mhead.file_type == Short3dSinogram || mhead.file_type == Float3dSinogram);
}


bool is_ECAT7_attenuation_file(const string& filename)
{
  Main_header mhead;
  return is_ECAT7_file(mhead, filename) &&
    mhead.file_type ==AttenCor;
}



void find_scanner(shared_ptr<Scanner> & scanner_ptr,const Main_header& mhead)
{  
  scanner_ptr = find_scanner_from_ECAT_system_type(mhead.system_type);
}


short find_ECAT_data_type(const NumericType& type, const ByteOrder& byte_order)
{
  if (!type.signed_type())
    warning("find_ECAT_data_type: CTI data support only signed types. Using the signed equivalent\n");
  if (type.integer_type())
  {
    switch(type.size_in_bytes())
    {
    case 1:
      return ByteData;
    case 2:
      return byte_order==ByteOrder::big_endian ? SunShort : VAX_Ix2;
    case 4:
      return byte_order==ByteOrder::big_endian ? SunLong : VAX_Ix4;
    default:
      {
        // write error message below
      }
    }
  }
  else
  {
    switch(type.size_in_bytes())
    {
    case 4:
      return byte_order==ByteOrder::big_endian ? IeeeFloat : VAX_Rx4;
    default:
      {
        // write error message below
      }
    }
  }
  string number_format;
  size_t size_in_bytes;
  type.get_Interfile_info(number_format, size_in_bytes);
  warning("find_ECAT_data_type: CTI does not support data type '%s' of %d bytes.\n",
          number_format.c_str(), size_in_bytes);
  return short(0);
}

/* -------------------------------------------
*	o f f s e t
* -------------------------------------------
*/
static long offset_in_ECAT_file (MatrixFile *mptr, int frame, int plane, int gate, int data,
            int bed, int segment, int *plane_size_ptr = NULL)
{
  
  int el_size[15], matnum, strtblk, group = abs(segment), i;
  long  plane_size = 0, off;
  struct MatDir matdir;
  Scan_subheader scansub;
  Image_subheader imagesub;
  Norm_subheader normsub;
  Attn_subheader attnsub;
  Scan3D_subheader scan3dsub;
  char *prog = "offset_in_ECAT_file";
  /*
  set_debug (prog);
  */
  el_size[ByteData] = 1;
  el_size[VAX_Ix2] = el_size[SunShort] = 2;
  el_size[VAX_Ix4] = el_size[VAX_Rx4] = el_size[IeeeFloat] = el_size[SunLong] = 4;
  
  if (mptr->mhptr->sw_version < V7)
    matnum = mat_numcod (frame, plane, gate, data, bed);
  else
    matnum = mat_numcod (frame, 1, gate, data, bed);
  print_debug (prog, "matnum = %d\n", matnum);
  
  if (matrix_find (mptr, matnum, &matdir) != 0)
    return -1;
  
  strtblk = matdir.strtblk;
  print_debug (prog, "strtblk = %d\n", strtblk);
  
  
  off = (strtblk + 1) * MatBLKSIZE;
  
  switch (mptr->mhptr->file_type)
  {
#ifndef STIR_NO_NAMESPACES
  case ::Sinogram:
#else
  case CTISinogram:
#endif

    {
      // KT 14/05/2002 added error check. 
      if (mat_read_scan_subheader (mptr->fptr, mptr->mhptr, strtblk, &scansub))
      {
        if (ferror(mptr->fptr))
          perror("offset_in_ECAT_file: error in reading subheader");
        return -1;
      }
      plane_size = scansub.num_r_elements *
        scansub.num_angles *
        el_size[scansub.data_type];
      
      if (mptr->mhptr->sw_version < V7)
        off = strtblk*MatBLKSIZE;
      else	   
        off = (strtblk + 1) * MatBLKSIZE + (plane-1) * plane_size;
      break;
    }
  case PetImage:
  case ByteVolume:
  case PetVolume:
    {
      // KT 14/05/2002 added error check.
      if (mat_read_image_subheader (mptr->fptr, mptr->mhptr, strtblk, &imagesub))
      {
        if (ferror(mptr->fptr))
          perror("offset_in_ECAT_file: error in reading subheader");
        return -1;
      }
      
      off = strtblk*MatBLKSIZE;
      plane_size = imagesub.x_dimension *
        imagesub.y_dimension *
        el_size[imagesub.data_type];
      
      if (mptr->mhptr->sw_version >= V7)
        off += (plane-1) * plane_size;
      break;
      
    }
    
  case AttenCor:   		
    {
      off = strtblk  * MatBLKSIZE;
      print_debug (prog, "off = %d\n", off);
      
      if (mptr->mhptr->sw_version >= V7)
      {
        print_debug (prog, "AttenCor\n");
      // KT 14/05/2002 added error check. 
      if (mat_read_attn_subheader (mptr->fptr, mptr->mhptr, strtblk, &attnsub))
      {
        if (ferror(mptr->fptr))
          perror("offset_in_ECAT_file: error in reading subheader");
        return -1;
      }
        
        
        switch (attnsub.storage_order)
        {
        case ElVwAxRd:
          plane_size = attnsub.num_r_elements *
            attnsub.num_angles *
            el_size[attnsub.data_type];
          
          if (group)
            for (i = 0; i < group; i++)
              off += plane_size * attnsub.z_elements[i];
	  // KT 25/10/2000 swapped segment order
	  if (segment > 0)
	    off += plane_size * attnsub.z_elements[group]/2;
	  off += (plane - 1) * plane_size;
	  break;
            
        case ElAxVwRd:
          print_debug (prog, "group %d, plane %d\n", group, plane);
          if (group)
            for (i = 0; i < group; i++)
            {
              plane_size = attnsub.num_r_elements *
                attnsub.z_elements[i] *
                el_size[attnsub.data_type];			    
              off += plane_size * attnsub.num_angles;
            }
            plane_size = attnsub.num_r_elements *
              attnsub.z_elements[group] *
              el_size[attnsub.data_type];
            if (group)
              plane_size /=2;
	    // KT 25/10/2000 swapped segment order
            if (segment > 0)
              off += plane_size*attnsub.num_angles;
            
            off += (plane - 1) *plane_size;
            
            break;
        }
        
      }
      break;
    }
    
    
  case Normalization:
    {
      
      // KT 14/05/2002 added error check. 
      if (mat_read_norm_subheader (mptr->fptr, mptr->mhptr, strtblk, &normsub))
      {
        if (ferror(mptr->fptr))
          perror("offset_in_ECAT_file: error in reading subheader");
        return -1;
      }
      off = strtblk*MatBLKSIZE;
      plane_size = normsub.num_r_elements *
        normsub.num_angles *
        el_size[normsub.data_type];
      if (mptr->mhptr->sw_version >= V7)
        off += (plane-1) * plane_size;
      break;
    }
  case ByteProjection:
  case PetProjection:
  case PolarMap:
    {
      fprintf (stderr, "Not implemented for this file type\n");
      off = -1;
      break;
    }
  case Byte3dSinogram:
  case Short3dSinogram:
  case Float3dSinogram :
    {
      off = (strtblk+1) * MatBLKSIZE;
      print_debug (prog, "off = %d\n", off);

      // KT 14/05/2002 added error check. 
      if (mat_read_Scan3D_subheader (mptr->fptr, mptr->mhptr, strtblk, &scan3dsub))
      {
        if (ferror(mptr->fptr))
          perror("offset_in_ECAT_file: error in reading subheader");
        return -1;
      }
      
      switch (scan3dsub.storage_order)
      {
      case ElVwAxRd:
        plane_size = scan3dsub.num_r_elements *
          scan3dsub.num_angles *
          el_size[scan3dsub.data_type];
        print_debug (prog, "xdim = %d (num_r_elements)\n", scan3dsub.num_r_elements);
        print_debug (prog, "ydim = %d (num_angles)    \n", scan3dsub.num_angles);
        print_debug (prog, "plane_size = %d \n", plane_size);
        if (group)
          for (i = 0; i < group; i++)
            off += (plane_size * scan3dsub.num_z_elements[i]);
	// KT 25/10/2000 swapped segment order
	if (segment > 0)
	  off += plane_size * scan3dsub.num_z_elements[group]/2;
	
          
	print_debug (prog, "num_z_elements[group] = %d\n", scan3dsub.num_z_elements[group]);
	print_debug (prog, "plane-1 = %d\n", plane-1);
	
	off += ((plane - 1) * plane_size);
	print_debug (prog, "off = %d\n", off);
	break;
          
      case ElAxVwRd:
        if (group)
          for (i = 0; i < group; i++)
          {
            plane_size = scan3dsub.num_r_elements *
              scan3dsub.num_z_elements[i] *
              el_size[scan3dsub.data_type];
            off += plane_size * scan3dsub.num_angles;
          }
          plane_size = scan3dsub.num_r_elements *
            scan3dsub.num_z_elements[group] *
            el_size[scan3dsub.data_type];
          // KT 14/05/2002 corrected. It seems that the convention of planes was different
          // now it's the same as for AttenCor
#if 0
          if (group)
          {
            plane_size /=2;
	    // KT 25/10/2000 swapped segment order
            if (segment > 0)
              off += plane_size;
            off += (plane - 1) *plane_size * 2;
          }
          else
            off += (plane - 1) *plane_size;
#else
          if (group)
              plane_size /=2;
	    // KT 25/10/2000 swapped segment order
          if (segment > 0)
            off += plane_size*scan3dsub.num_angles;
          off += (plane - 1) *plane_size;
#endif
          break;
      }
      break;
    }
  case Norm3d:
    {
      fprintf (stderr, "Not implemented yet\n");
      off = 1;
      break;
    }
  }
  
  if (plane_size_ptr != NULL)
    *plane_size_ptr = plane_size;
  
  return (off);

}


static void fill_string (char *str, int len)
{
    for (int i=0; i<len-2; i++) str[i]='.';
    str[len-2]='\0';
}

void make_ECAT7_main_header(Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name                     
                            )
{
  // first set to default (sometimes nonsensical) values
  strcpy(mhead.magic_number, "MATRIX7.0"); // TODO check
  fill_string(mhead.original_file_name, 32);
  mhead.sw_version= V7;
  mhead.system_type= -1;
  mhead.file_type= -1;
  fill_string(mhead.serial_number,10);
  mhead.scan_start_time= 0;
  fill_string(mhead.isotope_code, 8);
  mhead.isotope_halflife= 0.F;
  fill_string(mhead.radiopharmaceutical, 32);
  mhead.gantry_tilt= 0.F;
  mhead.gantry_rotation= 0.F;
  mhead.bed_elevation= 0.F;
  mhead.intrinsic_tilt = 0;
  mhead.wobble_speed= 0;
  mhead.transm_source_type= -1;
  mhead.distance_scanned = -1.F;
  mhead.transaxial_fov= -1.F;
  mhead.angular_compression = -1;
  mhead.calibration_factor= 0.F;
  mhead.calibration_units = 0;
  mhead.calibration_units_label = 0;
  mhead.compression_code = 0;
  fill_string(mhead.study_name, 12);
  fill_string(mhead.patient_id, 16);
  fill_string(mhead.patient_name, 32);
  mhead.patient_sex[0] = '.';
  mhead.patient_dexterity[0] = '.';
  mhead.patient_age = 0.F;
  mhead.patient_height = 0.F;
  mhead.patient_weight = 0.F;
  mhead.patient_birth_date=1;
  fill_string(mhead.physician_name,32);
  fill_string(mhead.operator_name,32);
  fill_string(mhead.study_description,32);
  mhead.acquisition_type = 0; 
  mhead.coin_samp_mode = 0; // default to net_trues
  mhead.axial_samp_mode= 0;
  mhead.patient_orientation = HeadFirstSupine;
  fill_string(mhead.facility_name, 20);
  mhead.num_planes= 0;
  mhead.num_frames= 1; // used for matnum, so set coherent default values
  mhead.num_gates= 1;
  mhead.num_bed_pos= 0;
  mhead.init_bed_position= -1.F;
  for (int i=0; i<15; i++) mhead.bed_offset[i]= 0.F;
  mhead.plane_separation= -1.F;
  mhead.lwr_sctr_thres= 0; // WARNING: default setup for the 966  
  mhead.lwr_true_thres= 350; // WARNING: default setup for the 966  
  mhead.upr_true_thres= 650; // WARNING: default setup for the 966  
  fill_string(mhead.user_process_code, 10);
  mhead.acquisition_mode= 0; // default to NORMAL
  mhead.bin_size = -1.F;
  mhead.branching_fraction = -1.F;
  mhead.dose_start_time = 0;
  mhead.dosage = 0.F;
  mhead.well_counter_factor = 1.F;
  fill_string(mhead.data_units,32);
  mhead.septa_state= -1;
  
  // now fill in what we can
  mhead.calibration_factor = 1.F;
  mhead.well_counter_factor=1.F;

  strncpy(mhead.original_file_name, orig_name.c_str(), 31);
  mhead.original_file_name[31]='\0';
  mhead.num_frames= 1; 
  
  mhead.system_type= find_ECAT_system_type(scanner);
  mhead.transaxial_fov= 
    scanner.get_inner_ring_radius()*2*
    sin(_PI/scanner.get_num_detectors_per_ring()*
	scanner.get_max_num_non_arccorrected_bins()/2.)/10;
  mhead.intrinsic_tilt = scanner.get_default_intrinsic_tilt();
  mhead.bin_size = scanner.get_default_bin_size()/10;
  mhead.plane_separation= scanner.get_ring_spacing()/2/10;
  mhead.intrinsic_tilt = scanner.get_default_intrinsic_tilt();
  
  mhead.distance_scanned=
    mhead.plane_separation * scanner.get_num_rings()*2;
}

void make_ECAT7_main_header(Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name,
                            DiscretisedDensity<3,float> const & density
                            )
{
  make_ECAT7_main_header(mhead, scanner, orig_name);
  
  DiscretisedDensityOnCartesianGrid<3,float> const & image =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(density);

  
  // extra main parameters that depend on data type
  mhead.file_type= PetVolume;
  mhead.num_planes=image.get_z_size();
  mhead.plane_separation=image.get_grid_spacing()[1]/10; // convert to cm

}

static short find_angular_compression(const ProjDataInfo& proj_data_info)
{
  // try to convert to cylindrical ProjDataInfo
  // use pointer such that we can check if it worked (without catching exceptions)
  ProjDataInfoCylindrical const * const proj_data_info_cyl_ptr =
    dynamic_cast<ProjDataInfoCylindrical const *>(&proj_data_info);
  if (proj_data_info_cyl_ptr!=0)
  {
    const int mash_factor =
      proj_data_info_cyl_ptr->get_view_mashing_factor();
    if (mash_factor>1 && mash_factor%2==1)
      {
	warning("ECAT7::find_angular_compression: odd mash factor %d is not supported by CTI header. "
		"Using a value of 0\n", mash_factor);
	return static_cast<short>(0);
      }
    else
      return static_cast<short>(mash_factor/2);
  }
  else
    {
      warning("ECAT7::find_angular_compression: proj data info does not correspond to a cylindrical scanner. "
	      "Using a value of 0\n");
      return static_cast<short>(0);
    }

}

static short find_axial_compression(const ProjDataInfo& proj_data_info)
{
  int axial_compression = 0;
  // try to convert to cylindrical ProjDataInfo
  // use pointer such that we can check if it worked (without catching exceptions)
  ProjDataInfoCylindrical const * const proj_data_info_cyl_ptr =
    dynamic_cast<ProjDataInfoCylindrical const *>(&proj_data_info);
  if (proj_data_info_cyl_ptr!=0)
  {
    axial_compression =
      proj_data_info_cyl_ptr->get_max_ring_difference(0) - 
      proj_data_info_cyl_ptr->get_min_ring_difference(0) + 1;
    for (int segment_num = proj_data_info.get_min_segment_num();
         segment_num <= proj_data_info.get_max_segment_num();
         ++segment_num)
    {
      const int this_segments_axial_compression =
          proj_data_info_cyl_ptr->get_max_ring_difference(segment_num) - 
          proj_data_info_cyl_ptr->get_min_ring_difference(segment_num) + 1;
      if (axial_compression != this_segments_axial_compression)
        error("ECAT 7 file format does not support data with non-uniform angular compression. "
              "Segment %d has angular compression %d while segment 0 has %d\n",
              segment_num, this_segments_axial_compression, axial_compression);
    }
  }
  else
  {
    axial_compression = 1;
    warning("ECAT 7 file format used with non-cylindrical ProjDataInfo type. "
            "I set axial_compression to 1, but who knows what will happen?");
  }
  return static_cast<short>(axial_compression);
}


NumericType 
make_ECAT7_main_header(Main_header& mhead,
		       const string& orig_name,
		       ProjDataInfo const & proj_data_info,
		       const bool write_as_attenuation,
		       NumericType output_type
		       )
{
  
  make_ECAT7_main_header(mhead, *proj_data_info.get_scanner_ptr(), orig_name);
  
  // extra main parameters that depend on data type
  
  mhead.num_planes = 0;
  for(int segment_num=proj_data_info.get_min_segment_num();
      segment_num <= proj_data_info.get_max_segment_num();
      ++segment_num)
    mhead.num_planes+= proj_data_info.get_num_axial_poss(segment_num);
  

  mhead.bin_size =proj_data_info.get_sampling_in_s(Bin(0,0,0,0))/10;

  mhead.angular_compression = find_angular_compression(proj_data_info);
  // guess septa state
  // assume that if it has more than 1 segment, it's a 3D scan...
  // except for some scanners without septa
  switch(proj_data_info.get_scanner_ptr()->get_type())
    {
    case Scanner::E966:
    case Scanner::E925:
    case Scanner::RATPET:
      mhead.septa_state= NoSeptaInstalled;
      break;
    default:
      mhead.septa_state= 
	proj_data_info.get_num_segments()==1 
	? SeptaExtended
	: SeptaRetracted;
    }


  if (write_as_attenuation)
    {
      mhead.file_type = AttenCor;
      mhead.acquisition_type = TransmissionScan;
      if (output_type != NumericType::FLOAT)
	{
	  warning("make_ECAT7_main_header: attenuation file will be written as floats "
		  "to avoid problems with CTI utilities");
	  output_type = NumericType::FLOAT;
	}
    
    }
  else
    {
      mhead.acquisition_type = StaticEmission;
      switch (output_type.id)
	{
	case NumericType::FLOAT:	  
	  mhead.file_type = Float3dSinogram; break;
	case NumericType::SHORT:
	  mhead.file_type = Short3dSinogram; break;
	case NumericType::SCHAR:
	  mhead.file_type = Byte3dSinogram; break;
	default:
	  warning("make_ECAT7_main_header: output type is not supported by ECAT7 format. Will use floats");
	  mhead.file_type = Float3dSinogram;
	  output_type = NumericType::FLOAT; 
	  break;
	}
    }

  return output_type;
}

// A utility function only called by scan_subheader_zero_fill
/*
  \internal 

  Most of the names of the variables we need are the same in the 
  Scan3D or Attn subheader, except num_z_elements and span.
  So, instead of writing essentially the same function twice, we
  use a templated version. Note that this takes care of the
  different locations of the information in the subheaders,
  as only the name is used.
*/

template <typename Subheader>
static void scan_subheader_zero_fill_aux(Subheader& shead) 
{ 
  shead.data_type= -1;
  shead.num_dimensions = -1;
  shead.num_r_elements = -1;
  shead.num_angles = -1;
  shead.ring_difference = -1;
  shead.storage_order = -1;
  shead.x_resolution = -1.F;
  shead.z_resolution = -1.F;
  shead.w_resolution = -1.F;
  shead.scale_factor= -1.F;
}

void scan_subheader_zero_fill(Scan3D_subheader& shead) 
{ 
  scan_subheader_zero_fill_aux(shead);
  shead.v_resolution = -1.F;
  shead.corrections_applied = 0;
  for (int i=0; i<64; ++i) shead.num_z_elements[i] = -1;
  shead.axial_compression = -1;
  shead.gate_duration= 0;
  shead.r_wave_offset= -1;
  shead.num_accepted_beats = -1;
  shead.scan_min= -1;
  shead.scan_max= -1;
  shead.prompts= -1;
  shead.delayed= -1;
  shead.multiples= -1;
  shead.net_trues= -1;
  shead.tot_avg_cor= -1.F;
  shead.tot_avg_uncor= -1.F;
  shead.total_coin_rate= -1;
  shead.frame_start_time= 0;
  shead.frame_duration= 0;
  shead.loss_correction_fctr = -1.F;
  for(int i=0;i<128;++i) shead.uncor_singles[i] = -1.F;

}

void scan_subheader_zero_fill(Attn_subheader& shead) 
{
  scan_subheader_zero_fill_aux(shead);
  shead.y_resolution = -1.F;
  shead.attenuation_type = 1; // default to measured
  shead.num_z_elements = -1;
  for (int i=0; i<64; ++i) shead.z_elements[i] = -1;
  shead.span = -1;
  shead.x_offset = -1.F;
  shead.y_offset = -1.F;
  shead.x_radius = -1.F;
  shead.y_radius = -1.F;
  shead.tilt_angle = -1.F;
  shead.attenuation_coeff = -1.F;
  shead.attenuation_min = -1.F;
  shead.attenuation_max = -1.F;
  shead.skull_thickness = -1.F;
  shead.num_additional_atten_coeff = -1;
  for (int i=0; i<8; ++i) shead.additional_atten_coeff[i] = -1.F;
  shead.edge_finding_threshold = -1.F;
}

void img_subheader_zero_fill(Image_subheader & ihead) 
{
  ihead.data_type= -1;
  ihead.num_dimensions= 3;
  ihead.x_dimension= -1;
  ihead.y_dimension= -1;
  ihead.z_dimension= -1;
  ihead.x_offset= 0.F;
  ihead.y_offset= 0.F;
  ihead.z_offset= 0.F;
  ihead.recon_zoom= -1.F;                    
  ihead.scale_factor= -1.F;
  ihead.image_min= -1;
  ihead.image_max= -1;
  ihead.x_pixel_size= -1.F;
  ihead.y_pixel_size= -1.F;
  ihead.z_pixel_size= -1.F;
  ihead.frame_duration= 0;
  ihead.frame_start_time= 0;
  ihead.filter_code = -1;
  ihead.x_resolution = -1.F;
  ihead.y_resolution = -1.F;
  ihead.z_resolution = -1.F;
  ihead.num_r_elements = -1;
  ihead.num_angles = -1;
  ihead.z_rotation_angle = -1;
  ihead.decay_corr_fctr= -1.F;
  ihead.processing_code= -1;
  ihead.gate_duration = 0;
  ihead.r_wave_offset = -1;
  ihead.num_accepted_beats = -1;
  ihead.filter_cutoff_frequency = -1.F;
  ihead.filter_resolution = -1.F;
  ihead.filter_ramp_slope = -1.F;
  ihead.filter_order = -1;
  ihead.filter_scatter_fraction = -1.F;
  ihead.filter_scatter_slope = -1.F;
  ihead.mt_1_1 = -1.F;
  ihead.mt_1_2 = -1.F;
  ihead.mt_1_3 = -1.F;
  ihead.mt_2_1 = -1.F;
  ihead.mt_2_2 = -1.F;
  ihead.mt_2_3 = -1.F;
  ihead.mt_3_1 = -1.F;
  ihead.mt_3_2 = -1.F;
  ihead.mt_3_3 = -1.F;
  ihead.rfilter_cutoff = -1.F;
  ihead.rfilter_resolution = -1.F;
  ihead.rfilter_code = -1;
  ihead.rfilter_order = -1;
  ihead.zfilter_cutoff = -1.F;
  ihead.zfilter_resolution = -1.F;
  ihead.zfilter_code = -1;
  ihead.zfilter_order = -1;
  ihead.mt_1_4 = -1.F;
  ihead.mt_2_4 = -1.F;
  ihead.mt_3_4 = -1.F;
  ihead.scatter_type = -1;
  ihead.recon_type = -1;
  ihead.recon_views = -1;
  fill_string(ihead.annotation, 40);  
}



//! A utility function only called by make_subheader_for_ECAT7(..., ProjDataInfo&)
/*!
  \internal 

  Most of the names of the variables we need are the same in the 
  Scan3D or Attn subheader, except num_z_elements and span.
  So, instead of writing essentially the same function twice, we
  use a templated version. Note that this takes care of the
  different locations of the information in the subheaders,
  as only the name is used.

  Extra parameters are used when the names of the variables do not match.
*/
template <typename SUBHEADERPTR>
static void
make_subheader_for_ECAT7_aux(SUBHEADERPTR sub_header_ptr, 
			  short * num_z_elements,
			  short& span,
			  const Main_header& mhead,
                          const ProjDataInfo& proj_data_info
                          )
{  
  scan_subheader_zero_fill(*sub_header_ptr);
  sub_header_ptr->num_dimensions = 4;
  sub_header_ptr->num_r_elements = proj_data_info.get_num_tangential_poss();
  sub_header_ptr->num_angles = proj_data_info.get_num_views();
  
  if (proj_data_info.get_max_segment_num() != 
      -proj_data_info.get_min_segment_num())
    error("ECAT 7 file format can only handle data with max_segment_num == -min_segment_num\n");
  span = find_axial_compression(proj_data_info);
  
  if (proj_data_info.get_max_segment_num() > 64)
    error("ECAT 7 file format supports only a maximum segment number of 64 while this data has %d\n",
          proj_data_info.get_max_segment_num());
  num_z_elements[0] = static_cast<short>(proj_data_info.get_num_axial_poss(0));
  for (int segment_num=1; segment_num<=proj_data_info.get_max_segment_num(); ++segment_num)
  {
    num_z_elements[segment_num] =
      static_cast<short>(2*proj_data_info.get_num_axial_poss(segment_num));
  }
  for (int i=proj_data_info.get_max_segment_num()+1; i<64; ++i) 
    num_z_elements[i] = 0;
  
  // try to convert to cylindrical ProjDataInfo
  // use pointer such that we can check if it worked (without catching exceptions)
  const ProjDataInfoCylindrical * const proj_data_info_cyl_ptr =
    dynamic_cast<ProjDataInfoCylindrical const * const>(&proj_data_info);
  if (proj_data_info_cyl_ptr!=0)
  {
    sub_header_ptr->ring_difference =
      proj_data_info_cyl_ptr->get_max_ring_difference(proj_data_info.get_max_segment_num());    
  }
  else
  {
    sub_header_ptr->ring_difference = -1;    
  }
  
  sub_header_ptr->x_resolution = proj_data_info.get_sampling_in_s(Bin(0,0,0,0))/10;
  sub_header_ptr->storage_order = ElAxVwRd;

  
}


// WARNING data_type has still to be set
void
make_subheader_for_ECAT7(Attn_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         )
{
  make_subheader_for_ECAT7_aux(&shead, shead.z_elements, shead.span,
                     mhead, proj_data_info);
  if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const * const>(&proj_data_info))
  {
    warning("make_subheader_for_ECAT7: data is not arc-corrected but info is not available in CTI attenuation subheader\n");
  }
}
    
// WARNING data_type has to be set
void
make_subheader_for_ECAT7(Scan3D_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         )
{
  make_subheader_for_ECAT7_aux(&shead, shead.num_z_elements, shead.axial_compression,
                     mhead, proj_data_info);
  // try to convert to cylindrical ProjDataInfo to check if it's arccorrected
  // use pointer such that we can check if it worked (without catching exceptions)
  if (dynamic_cast<ProjDataInfoCylindricalArcCorr const * const>(&proj_data_info))
  {
    shead.corrections_applied = static_cast<short>(ArcPrc);
  }
  else if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const * const>(&proj_data_info))
  {
    shead.corrections_applied = 0;
  }
  else
  {
    warning("make_subheader_for_ECAT7: unknown type of proj_data_info. Setting data to arc-corrected anyway\n");
    shead.corrections_applied = static_cast<short>(ArcPrc);
  }
  
}


//! A utility function only called by make_pdfs_matrix()
/*!
  \internal 

  Most of the names of the variables we need are the same in the 
  Scan3D or Attn subheader, except num_z_elements and span.
  So, instead of writing essentially the same function twice, we
  use a templated version. Note that this takes care of the
  different locations of the information in the subheaders,
  as only the name is used.

  Extra parameters are used when the names of the variables do not match.
*/
template <typename SUBHEADERPTR>
static
ProjDataFromStream * 
make_pdfs_from_matrix_aux(SUBHEADERPTR sub_header_ptr, 
			  short const * num_z_elements,
			  const int span,
			  const bool arc_corrected,
			  MatrixFile * const mptr, 
			  MatrixData * const matrix, 
			  const shared_ptr<iostream>&  stream_ptr)
{
  shared_ptr<Scanner> scanner_ptr;
  find_scanner(scanner_ptr, *(mptr->mhptr)); 
  if (scanner_ptr->get_type() == Scanner::Unknown_scanner)
  {
    warning("ECAT7 IO: Couldn't determine the scanner \n"
      "(Main_header.system_type=%d), defaulting to 962.\n"
      "This might give dramatic problems.\n",
	    mptr->mhptr->system_type); 
    scanner_ptr = new Scanner(Scanner::E962);
  }
#ifdef B_JOINT_STIRGATE
  // zlong, 08-04-2004, add support for Unknown_scanner
  // we have no idea about the geometry, so, ask user.
  if(scanner_ptr->get_type() == Scanner::Unknown_scanner)
  {
    warning("Joint Gate Stir project warning:\n");
    warning("I have no idea about your scanner, please give me the scanner info.\n");
    scanner_ptr = Scanner::ask_parameters();
  }
#endif

 
  if(sub_header_ptr->num_dimensions != 4)
    warning("Expected subheader.num_dimensions==4. Continuing...\n");
  const int num_tangential_poss = sub_header_ptr->num_r_elements;
  const int num_views = sub_header_ptr->num_angles;

  // find maximum segment
  int max_segment_num = 0;
  while(max_segment_num<64 && num_z_elements[max_segment_num+1] != 0)
    ++max_segment_num;
  
  VectorWithOffset<int> num_axial_poss_per_seg(-max_segment_num,max_segment_num);
  
  num_axial_poss_per_seg[0] = num_z_elements[0];
  for (int segment_num=1; segment_num<=max_segment_num; ++segment_num)
  {
    num_axial_poss_per_seg[-segment_num] =
      num_axial_poss_per_seg[segment_num] = 
      num_z_elements[segment_num]/2;
  }
  
  const int max_delta = sub_header_ptr->ring_difference;
  const float bin_size = sub_header_ptr->x_resolution * 10; // convert to mm
  const float scale_factor = sub_header_ptr->scale_factor;
  
  ProjDataFromStream::StorageOrder storage_order;
  switch (sub_header_ptr->storage_order)
  {
  case ElVwAxRd:
    storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
    break;
    
  case ElAxVwRd:
    storage_order = ProjDataFromStream::Segment_View_AxialPos_TangPos;
    break;
  default:
    warning("Funny value for subheader.storage_order. Assuming ElVwAxRd\n");
    storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
  }
  NumericType data_type;
  ByteOrder byte_order;
  find_type_from_ECAT_data_type(data_type, byte_order, sub_header_ptr->data_type);

  if (fabs(bin_size - scanner_ptr->get_default_bin_size())>.01)
  {
    warning("Bin size from header (%g) does not agree with expected value %g\nfor scanner %s. Using bin size from header...\n",
	    bin_size, 
	    scanner_ptr->get_default_bin_size(), 
	    scanner_ptr->get_name().c_str());
    scanner_ptr->set_default_bin_size(bin_size);
  }
  // TODO more checks on FOV etc.

  int span_to_use = span;
  if (span == 0)
    {
      if (num_z_elements[0] == scanner_ptr->get_num_rings())
	{
	  warning("\nECAT7 subheader says span=0, while span should be odd.\n"
		  "However, num_z_elements[0]==num_rings, so we'll asssume it's span=1\n");
	  span_to_use = 1;
	}
      else
	{
	  error("\nECAT7 subheader says span=0, while span should be odd.\n"
		"Moreover,  num_z_elements[0] (%d)!=num_rings (%d), so, I give up.\n",
		num_z_elements[0], scanner_ptr->get_num_rings());
	}
    }
  
  shared_ptr<ProjDataInfo> pdi_ptr =
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, span_to_use, max_delta, 
				  num_views, num_tangential_poss,  
				  arc_corrected);
  
  pdi_ptr->set_num_axial_poss_per_segment(num_axial_poss_per_seg);
    
  vector<int> segment_sequence_in_stream(2*max_segment_num+1);
  // KT 25/10/2000 swapped segment order
  // ECAT 7 always stores segments as 0, -1, +1, ...
  segment_sequence_in_stream[0] = 0;
  for (int segment_num = 1; segment_num<=max_segment_num; ++segment_num)
  {
    segment_sequence_in_stream[2*segment_num-1] = -segment_num;
    segment_sequence_in_stream[2*segment_num] = segment_num;
  }
  
  Matval matval;
  mat_numdoc(matrix->matnum, &matval);

  const long offset_in_file =
    offset_in_ECAT_file(mptr,
			matval.frame, 1, matval.gate, matval.data, matval.bed,
			0, NULL);
  // KT 14/05/2002 added error check
  if (offset_in_ECAT_file<0)
    return 0;

  
  return new ProjDataFromStream (pdi_ptr, stream_ptr, offset_in_file, 
				 segment_sequence_in_stream,
				 storage_order,
				 data_type,
				 byte_order,
				 scale_factor);
}


ProjDataFromStream * 
make_pdfs_from_matrix(MatrixFile * const mptr, 
                      MatrixData * const matrix, 
                      const shared_ptr<iostream>&  stream_ptr)
{
  switch (mptr->mhptr->file_type)
    {
    case AttenCor:   		
      {
	Attn_subheader const *sub_header_ptr= 
	  reinterpret_cast<Attn_subheader const*>(matrix->shptr);
	
	// CTI does not provide corrections_applied to check if the data
	// is arc-corrected. Presumably it's attenuation data is always 
	// arccorrected
	const bool arc_corrected = true;
	warning("Assuming data is arc-corrected (info not available in CTI attenuation subheader)\n");
	return   
	  make_pdfs_from_matrix_aux(sub_header_ptr, 
				    sub_header_ptr->z_elements,
				    sub_header_ptr->span,
				    arc_corrected,
				    mptr, matrix, stream_ptr);
      }
    case Byte3dSinogram:
    case Short3dSinogram:
    case Float3dSinogram :
      {
	Scan3D_subheader const * sub_header_ptr= 
	  reinterpret_cast<Scan3D_subheader const*>(matrix->shptr);

	ProcessingCode cti_processing_code =
	  static_cast<ProcessingCode>(sub_header_ptr->corrections_applied);
	
	const bool arc_corrected = 
	  (cti_processing_code & ArcPrc) != 0;
	
	return   
	  make_pdfs_from_matrix_aux(sub_header_ptr, 
				    sub_header_ptr->num_z_elements,
				    sub_header_ptr->axial_compression,
				    arc_corrected,
				    mptr, matrix, stream_ptr);
      }
    default:
      {
	warning ("make_pdfs_from_matrix: unsupported file_type %d\n",
	       mptr->mhptr->file_type);
	return NULL;
      }
    }
}

static
Succeeded
get_ECAT7_image_info(CartesianCoordinate3D<int>& dimensions,
		     CartesianCoordinate3D<float>& voxel_size,
		     Coordinate3D<float>& origin,
		     float& scale_factor,      
		     NumericType& type_of_numbers,
		     ByteOrder& byte_order,
		     long& offset_in_file,

		     const string& ECAT7_filename,
		     const int frame_num, const int gate_num, const int data_num, const int bed_num,
		     const char * const warning_prefix,
		     const char * const warning_suffix)
{
  MatrixFile * const mptr = 
    matrix_open( ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror( ECAT7_filename.c_str());
    return Succeeded::no;
  }
  if (mptr->mhptr->sw_version < V7)
    { 
      matrix_close(mptr);
      warning("%s: %s seems to be an ECAT 6 file. "
	      "%s",
	      warning_prefix, ECAT7_filename.c_str(), warning_suffix); 
      return Succeeded::no;
    }

  //case PetImage: TODO this probably has subheaders?
  if (mptr->mhptr->file_type != ByteVolume &&
      mptr->mhptr->file_type != PetVolume)
    {
      matrix_close(mptr);
      warning("%s: %s has the wrong file type to be read as an image."
	      "%s",
	      warning_prefix, ECAT7_filename.c_str(), warning_suffix); 
      return Succeeded::no;
    }
  
  const int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
  MatrixData* matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
  
  if (matrix==NULL)
    { 
      matrix_close(mptr);
      warning("%s: Matrix not found at \"%d,1,%d,%d,%d\" in file %s\n."
	      "%s",
	      warning_prefix,
	      frame_num, gate_num, data_num, bed_num, ECAT7_filename.c_str(),
	      warning_suffix);
      return Succeeded::no;
    }  

  Image_subheader const * const sub_header_ptr=
    reinterpret_cast<Image_subheader const* const>(matrix->shptr);
    
  if(sub_header_ptr->num_dimensions != 3)
    warning("%s: while reading matrix \"%d,1,%d,%d,%d\" in file %s:\n"
	    "Expected subheader_ptr->num_dimensions==3. Continuing\n",
	    warning_prefix,
	    frame_num, gate_num, data_num, bed_num, ECAT7_filename.c_str());
  dimensions = 
    CartesianCoordinate3D<int>(matrix->zdim,
		      matrix->ydim,
		      matrix->xdim);
  voxel_size =
    CartesianCoordinate3D<float>(matrix->z_size * 10,
				 matrix->y_size * 10,
				 matrix->pixel_size * 10); // convert to mm

  origin =
    Coordinate3D<float>(matrix->z_origin * 10,
			matrix->y_origin * 10,
			matrix->x_origin * 10); // convert to mm
      
  scale_factor = matrix->scale_factor;
      
  find_type_from_ECAT_data_type(type_of_numbers, byte_order, matrix->data_type);
      
  offset_in_file =
    offset_in_ECAT_file(mptr, frame_num, 1, gate_num, data_num, bed_num, 0, NULL);
  if (offset_in_ECAT_file<0)
    { 
      free_matrix_data(matrix);
      matrix_close(mptr);
      warning("%s: while reading matrix \"%d,1,%d,%d,%d\" in file %s:\n"
	      "Error in determining offset into ECAT7 file %s.\n"
	      "%s",
	      warning_prefix, 
	      frame_num, gate_num, data_num, bed_num, ECAT7_filename.c_str(),
	      warning_suffix);
      return Succeeded::no;
    }     
    
  free_matrix_data(matrix);
  matrix_close(mptr);
  return Succeeded::yes;
}

VoxelsOnCartesianGrid<float> *
ECAT7_to_VoxelsOnCartesianGrid(const string& ECAT7_filename,
			       const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  const char * const warning_prefix = "ECAT7_to_VoxelsOnCartesianGrid";
  const char * const warning_suffix =  "I'm not reading any data...\n";

  CartesianCoordinate3D<int> dimensions;
  CartesianCoordinate3D<float> voxel_size;
  Coordinate3D<float> origin;
  float scale_factor;
  NumericType type_of_numbers;
  ByteOrder byte_order;
  long offset_in_file;
  if (get_ECAT7_image_info(dimensions, voxel_size, origin,
			   scale_factor, type_of_numbers, byte_order, offset_in_file,

			   ECAT7_filename,
			   frame_num, gate_num, data_num, bed_num,
			   warning_prefix,
			   warning_suffix) ==
      Succeeded::no)
    { 
      return 0;
    }

  const IndexRange3D range_3D (0,dimensions.z()-1,
			       -dimensions.y()/2,(-dimensions.y()/2)+dimensions.y()-1,
			       -dimensions.x()/2,(-dimensions.x()/2)+dimensions.x()-1);
  VoxelsOnCartesianGrid<float>* image_ptr =
    new VoxelsOnCartesianGrid<float> (range_3D, origin, voxel_size);
  
  std::ifstream data_in(ECAT7_filename.c_str(), ios::in | ios::binary);
  if (!data_in)
    {
      warning("%s: cannot open %s using C++ ifstream.\n"
	      "%s",  
	      warning_prefix, ECAT7_filename.c_str(), warning_suffix); 
      return 0;
    }

  data_in.seekg(static_cast<unsigned long>(offset_in_file));
  if (!data_in)
    {
      warning("%s: while reading %s:\n"
	      "error seeking to position of data.\n"
	      "%s",  
	      warning_prefix, ECAT7_filename.c_str(), warning_suffix); 

      return 0;
    }
  
  {
    float scale = float(1);
    image_ptr->read_data(data_in, type_of_numbers, scale, byte_order);
    if (scale != 1)
      {
	warning("%s: while reading %s:\n"
		"error in reading data with convertion to floats.\n",
		"%s", 
		warning_prefix, ECAT7_filename.c_str(), warning_suffix); 

	return 0;
      }
  }
  *image_ptr *= scale_factor;
    
  return image_ptr;
}

ProjDataFromStream*
ECAT7_to_PDFS(const string& ECAT7_filename,
	      const int frame_num, const int gate_num, const int data_num, const int bed_num)
{  
  MatrixFile * const mptr = matrix_open( ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror( ECAT7_filename.c_str());
    return 0;
  }
  const char * const warning_prefix = "ECAT7_to_PDFS";
  const char * const warning_suffix =  "I'm not reading any data...\n";

  if (mptr->mhptr->sw_version < V7)
    { 
      warning("%s: %s seems to be an ECAT 6 file. "
	      "%s",  warning_prefix, ECAT7_filename.c_str(), warning_suffix); 
      return 0; 
    }
  const int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
  MatrixData* matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
  
  if (matrix==NULL)
    { 
      matrix_close(mptr);
      warning("%s: Matrix not found at \"%d,1,%d,%d,%d\" in file %s\n."
	      "%s",
	      warning_prefix,
	      frame_num, gate_num, data_num, bed_num, 
	      ECAT7_filename.c_str(), warning_suffix);
      return 0;
    }
  
  shared_ptr<iostream> stream_ptr = 
    new fstream(ECAT7_filename.c_str(), ios::in | ios::binary);
      
  ProjDataFromStream * pdfs_ptr = 
    make_pdfs_from_matrix(mptr, matrix, stream_ptr);
  free_matrix_data(matrix);
  matrix_close(mptr);
  return pdfs_ptr;
}


Succeeded 
write_basic_interfile_header_for_ECAT7(string& interfile_header_filename,
                                       const string& ECAT7_filename,
				       const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  
  MatrixFile * const mptr = matrix_open( ECAT7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror( ECAT7_filename.c_str());
    return Succeeded::no;
  }
  const char * const warning_prefix = "write_basic_interfile_header_for_ECAT7";
  const char * const warning_suffix =  "I'm not writing an Interfile header...\n";

  if (mptr->mhptr->sw_version < V7)
  { 
    warning("%s: '%s' seems to be an ECAT 6 file. "
            "%s",  warning_prefix, ECAT7_filename.c_str(), warning_suffix); 
    return Succeeded::no; 
  }
  

  char *header_filename = new char[ECAT7_filename.size() + 100];
  {
    strcpy(header_filename, ECAT7_filename.c_str());
    // keep extension, just in case we would have conflicts otherwise
    // but replace the . with a _
    char * dot_ptr = strchr(find_filename(header_filename),'.');
    if (dot_ptr != NULL)
      *dot_ptr = '_';
    // now add stuff to say which frame, gate, bed, data this was
    sprintf(header_filename+strlen(header_filename), "_f%dg%dd%db%d", 
	    frame_num, gate_num, data_num, bed_num);
  }
  
  switch (mptr->mhptr->file_type)
  {
  //case PetImage: // TODO this probably has subheaders?
  case ByteVolume:
  case PetVolume:
    {
      CartesianCoordinate3D<int> dimensions;
      CartesianCoordinate3D<float> voxel_size;
      Coordinate3D<float> origin;
      float scale_factor;
      NumericType type_of_numbers;
      ByteOrder byte_order;
      long offset_in_file;
      if (get_ECAT7_image_info(dimensions, voxel_size, origin,
			       scale_factor, type_of_numbers, byte_order, offset_in_file,

			       ECAT7_filename,
			       frame_num, gate_num, data_num, bed_num,
			       warning_prefix,
			       warning_suffix) ==
	  Succeeded::no)
	{ 
	  matrix_close(mptr);
	  return Succeeded::no;
	}
      
      VectorWithOffset<float> scaling_factors(1);
      VectorWithOffset<unsigned long> file_offsets(1);
      scaling_factors[0] = scale_factor;
      file_offsets[0] = static_cast<unsigned long>(offset_in_file);
      strcat(header_filename, ".hv");
      interfile_header_filename = header_filename;
      write_basic_interfile_image_header(header_filename, ECAT7_filename,
					 dimensions, voxel_size, type_of_numbers, byte_order,
					 scaling_factors,
					 file_offsets);
      break;
    }
          
  case AttenCor:   		
  case Byte3dSinogram:
  case Short3dSinogram:
  case Float3dSinogram :
    {
      const int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
      MatrixData* matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
  
      if (matrix==NULL)
	{ 
	  matrix_close(mptr);
	  warning("%s: Matrix not found at \"%d,1,%d,%d,%d\" in file '%s'.\n"
		  "%s",
		  warning_prefix,
		  frame_num, gate_num, data_num, bed_num, 
		  ECAT7_filename.c_str(), warning_suffix);
	  return Succeeded::no;
	}
  
      shared_ptr<iostream> stream_ptr = 
	new fstream(ECAT7_filename.c_str(), ios::in | ios::binary);
      
      shared_ptr<ProjDataFromStream> pdfs_ptr = 
	make_pdfs_from_matrix(mptr, matrix, stream_ptr);
      free_matrix_data(matrix);
     
      if (pdfs_ptr == NULL)
	{
	  matrix_close(mptr);
	  return Succeeded::no;
	}
      strcat(header_filename, ".hs");
      interfile_header_filename = header_filename;
      write_basic_interfile_PDFS_header(header_filename, ECAT7_filename, *pdfs_ptr);

      break;
    }    
        
  default:
    matrix_close(mptr);
    warning("%s: File type not handled for file '%s'.\n"
	    "%s",
	    warning_prefix, ECAT7_filename.c_str(), warning_suffix);
    return Succeeded::no;
  }
  
  delete header_filename;
  matrix_close(mptr);
  return Succeeded::yes;
}


Succeeded 
DiscretisedDensity_to_ECAT7(MatrixFile *mptr,
                            DiscretisedDensity<3,float> const & density, 
			    const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  
  const Main_header& mhead = *(mptr->mhptr);
  DiscretisedDensityOnCartesianGrid<3,float> const & image =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(density);

   
  if (mhead.file_type!= PetVolume)
  {
    warning("DiscretisedDensity_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.file_type should be ImageFile\n",
            frame_num, gate_num, data_num, bed_num);
    return Succeeded::no;
  }
  if (mhead.num_planes!=image.get_z_size())
  {
    warning("DiscretisedDensity_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.num_planes should be %d\n",
            frame_num, gate_num, data_num, bed_num,image.get_z_size());
    return Succeeded::no;
  }
  const float voxel_size_z = image.get_grid_spacing()[1]/10;// convert to cm
  const float voxel_size_y = image.get_grid_spacing()[2]/10;
  const float voxel_size_x = image.get_grid_spacing()[3]/10;
  if (fabs(mhead.plane_separation - voxel_size_z) > 1.E-4) 
  {
    warning("DiscretisedDensity_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.plane_separation should be %g\n",
            frame_num, gate_num, data_num, bed_num,voxel_size_z);
    return Succeeded::no;
  }
  
  
  Image_subheader ihead;
  img_subheader_zero_fill(ihead);
  
  const int z_size= image.get_z_size();
  const int y_size= image.get_y_size();
  const int x_size= image.get_x_size();
  
  // Setup subheader params
  // ihead.data_type set by save_volume7;
  ihead.x_dimension= x_size;
  ihead.y_dimension= y_size;
  ihead.z_dimension= z_size;
  ihead.x_pixel_size= voxel_size_x;
  ihead.y_pixel_size= voxel_size_y;
  ihead.z_pixel_size= voxel_size_z;
  
  ihead.num_dimensions= 3;
  ihead.x_offset= image.get_origin().x()/10;
  ihead.y_offset= image.get_origin().y()/10;
  ihead.z_offset= image.get_origin().z()/10;
  shared_ptr<Scanner> scanner_ptr;
  find_scanner(scanner_ptr, mhead);

  ihead.recon_zoom= 
    mhead.bin_size/voxel_size_x *
    scanner_ptr->get_default_num_arccorrected_bins()/128.F;

  ihead.decay_corr_fctr= 1;

#if 0  
  // attempt to write this ourselves, but we'd need to write the subheader, and we 
  // don't have neat functions for written Arrays to FILE anyway (only to streams)
  NumericType data_type;
  ByteOrder byte_order;
  find_type_from_ECAT_data_type(data_type, byte_order, mhead.data_type);
  
  const long offset_in_file =
    offset_in_ECAT_file(mptr, frame, 1, gate, data, bed, 0, NULL);
  
  // KT 14/05/2002 added error check
  if (offset_in_ECAT_file<0)
  { 
    warning("Error in determining offset into ECAT file for segment %d (f%d, g%d, d%d, b%d)\n"
      "No data written for this segment and all remaining segments\n",
      segment_num, frame_num, gate_num, data_num, bed_num);
    return Succeeded::no; 
  }  if (fseek(mptr->fptr, offset_in_file, SEEK_SET) !=0)
  {
    warning("DiscretisedDensity_to_ECAT7: could not seek to file position %d\n"
      "No data written for frame %d, gate %d, data %d, bed %d\n",
      offset_in_file,
            frame_num, gate_num,data_num, bed_num);
    return Succeeded::no;
  }
    //TODO somehow write data
  return Succeeded::no;
#else
  // use LLN function
  // easy, but wasteful: we need to copy the data first to a float buffer
  // then save_volume7 makes a short buffer...
  const unsigned int buffer_size = 
    static_cast<unsigned int>(x_size)*
    static_cast<unsigned int>(y_size)*
    static_cast<unsigned int>(z_size);
  auto_ptr<float> float_buffer =
     auto_ptr<float>(new float[buffer_size]);
  // save_volume7 does a swap in z, so we can't use the following
  //copy(density.begin_all(), density.end_all(), float_buffer.get());
  {
    float * current_buffer_pos = float_buffer.get();
    const unsigned int plane_size = 
      static_cast<unsigned int>(x_size)*
      static_cast<unsigned int>(y_size);
    for (int z=density.get_max_index(); z>= density.get_min_index(); --z)
    {
      copy(density[z].begin_all(), density[z].end_all(), current_buffer_pos);
      current_buffer_pos += plane_size;
    }
  }
  if (save_volume7(mptr, &ihead, float_buffer.get(),
                   frame_num, gate_num,data_num, bed_num) != 0)
  {
    warning("Error writing image to ECAT7 file.\n"
            "No data written for frame %d, gate %d, data %d, bed %d\n",
            frame_num, gate_num,data_num, bed_num);
    return Succeeded::no;
  }
  else
  {
    return Succeeded::yes;
  }
#endif
}


Succeeded 
DiscretisedDensity_to_ECAT7(DiscretisedDensity<3,float> const & density, 
			    string const & cti_name, string const&orig_name,
			    const Scanner& scanner,
                            const int frame_num, const int gate_num, const int data_num, const int bed_num)
{  
  Main_header mhead;
  make_ECAT7_main_header(mhead, scanner, orig_name, density);

  
  MatrixFile* mptr= matrix_create (cti_name.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    return Succeeded::no;
  Succeeded result =
    DiscretisedDensity_to_ECAT7(mptr,
                            density, 
			    frame_num, gate_num,data_num, bed_num);
  
  matrix_close(mptr);    
  return result;
}


Succeeded
update_ECAT7_subheader(MatrixFile *mptr, Scan_subheader& shead,
		       const MatDir& matdir)
{
  const int ERROR=-1;
  if (mptr->mhptr->file_type !=
#ifndef STIR_NO_NAMESPACES
      ::Sinogram
#else
      CTISinogram
#endif
      )
    return Succeeded::no;
  return
    mat_write_scan_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			     &shead) == ERROR ?
    Succeeded::no : Succeeded::yes;
}


Succeeded
update_ECAT7_subheader(MatrixFile *mptr, Norm_subheader& shead, const MatDir& matdir)
{
  const int ERROR=-1;
  if (mptr->mhptr->file_type != Normalization)
    return Succeeded::no;
  return
    mat_write_norm_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			     &shead) == ERROR ?
    Succeeded::no : Succeeded::yes;
}


Succeeded
update_ECAT7_subheader(MatrixFile *mptr, Image_subheader& shead, const MatDir& matdir)
{
  const int ERROR=-1;
  if (!(mptr->mhptr->file_type == PetImage  ||
	mptr->mhptr->file_type == ByteVolume  ||
	mptr->mhptr->file_type == PetVolume))
    return Succeeded::no;
  return
    mat_write_image_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			     &shead) == ERROR ?
    Succeeded::no : Succeeded::yes;
}

Succeeded
update_ECAT7_subheader(MatrixFile *mptr, Attn_subheader& shead, const MatDir& matdir)
{
  const int ERROR=-1;
  if (mptr->mhptr->file_type != AttenCor)
    return Succeeded::no;
  return
    mat_write_attn_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			     &shead) == ERROR ?
    Succeeded::no : Succeeded::yes;
}

Succeeded
update_ECAT7_subheader(MatrixFile *mptr, Scan3D_subheader& shead, const MatDir& matdir)
{
  const int ERROR=-1;
  if (!(mptr->mhptr->file_type == Byte3dSinogram  ||
	mptr->mhptr->file_type == Short3dSinogram  ||
	mptr->mhptr->file_type == Float3dSinogram))
    return Succeeded::no;
  return
    mat_write_Scan3D_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
			     &shead) == ERROR ?
    Succeeded::no : Succeeded::yes;
}

template <class SUBHEADER_TYPE>
Succeeded
update_ECAT7_subheader(MatrixFile *mptr, SUBHEADER_TYPE& shead,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  const int ERROR=-1;
    const int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
    MatDir matdir;
    if (matrix_find(mptr, matnum, &matdir) == ERROR)
      return  Succeeded::no;
    
    return update_ECAT7_subheader(mptr, shead, matdir);
}

namespace detail
{

template <class OutputType>
Succeeded 
static 
ProjData_to_ECAT7_help(MatrixFile *mptr, const NumericInfo<OutputType>& output_type_info,
		       ProjData const& proj_data, 
		       const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  const ByteOrder output_byte_order =
 //   output_type_info.integer_type() ? ByteOrder::get_native_order() : ByteOrder::big_endian;
ByteOrder::big_endian;

  const short int cti_data_type = 
    find_ECAT_data_type(output_type_info.type_id(), output_byte_order);
  if (cti_data_type==0)
    return Succeeded::no;
  const Main_header& mhead = *(mptr->mhptr);
  {
    int num_planes = 0;
    for(int segment_num=proj_data.get_min_segment_num();
        segment_num <= proj_data.get_max_segment_num();
        ++segment_num)
     num_planes+= proj_data.get_num_axial_poss(segment_num);
  
    if (mhead.num_planes!=num_planes) // TODO check if this is the usual convention
    {
      warning("ProjData_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
              "Main header.num_planes should be %d",
              frame_num, gate_num, data_num, bed_num,num_planes);
      return Succeeded::no;
    }
  }  
  
  // find scale factor in case we're not writing floats
  float scale_factor = 1;
  if (output_type_info.integer_type())
    {
      scale_factor = 0;// set first to 0 to use maximum range of output type
      for(int segment_num=proj_data.get_min_segment_num();
	  segment_num <= proj_data.get_max_segment_num();
	  ++segment_num)
	{    
	  const SegmentByView<float> segment = 
	    proj_data.get_segment_by_view(segment_num);

	  find_scale_factor(scale_factor,
			    segment, 
			    NumericInfo<OutputType>());
	}
    }
  cout << "\nProjData_to_ECAT7: Will use scale factor " << scale_factor;

  Scan3D_subheader scan3d_shead; 
  Attn_subheader attn_shead;

  if (mhead.file_type == AttenCor)
  {
    make_subheader_for_ECAT7(attn_shead, mhead, *proj_data.get_proj_data_info_ptr());
    // Setup remaining subheader params
    attn_shead.data_type= cti_data_type;
    attn_shead.scale_factor= scale_factor; 
    attn_shead.storage_order = ElAxVwRd;
  }
  else
  {
    make_subheader_for_ECAT7(scan3d_shead, mhead, *proj_data.get_proj_data_info_ptr());
    // Setup remaining subheader params
    scan3d_shead.data_type= cti_data_type;
    scan3d_shead.loss_correction_fctr= 1.F; 
    scan3d_shead.scale_factor= scale_factor; 
    scan3d_shead.storage_order = ElAxVwRd;
  }

  // allocate space in file, and write subheader
  {
    const int ERROR=-1;
    const int plane_size= proj_data.get_num_tangential_poss() * proj_data.get_num_views();
    
    // TODO only ok if main_header.num_planes is set as above
    int nblks = (mhead.num_planes*plane_size*sizeof(OutputType)+511)/512;
    
    /* 3D sinograms subheader use one more block */
    if (mptr->mhptr->file_type == Byte3dSinogram  ||
      mptr->mhptr->file_type == Short3dSinogram  ||
      mptr->mhptr->file_type == Float3dSinogram) nblks += 1;
    
    int matnum = mat_numcod (frame_num, 1, gate_num, data_num, bed_num);
    struct MatDir matdir;
    if (matrix_find(mptr, matnum, &matdir) == ERROR)
    {
      int blkno = mat_enter(mptr->fptr, mptr->mhptr, matnum, nblks) ;
      if( blkno == ERROR ) return( Succeeded::no );
      matdir.matnum = matnum ;
      matdir.strtblk = blkno ;
      matdir.endblk = matdir.strtblk + nblks - 1 ;
      matdir.matstat = 1 ;
      insert_mdir(matdir, mptr->dirlist) ;
    }
    
    if (mhead.file_type == AttenCor)
    {
      if (mat_write_attn_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
                                   &attn_shead) == ERROR) 
         return  Succeeded::no;
    }
    else
    {
      if (mat_write_Scan3D_subheader(mptr->fptr, mptr->mhptr, matdir.strtblk,
                                   &scan3d_shead) == ERROR) 
         return  Succeeded::no;
    }
    
  }

  cout<<"\nProcessing segment number:";
  
  for(int segment_num=proj_data.get_min_segment_num();
      segment_num <= proj_data.get_max_segment_num();
      ++segment_num)
  {    
    cout<<"  "<<segment_num;

    // read the segment
    const SegmentByView<float> segment = 
      proj_data.get_segment_by_view(segment_num);

    const long offset_in_file =
      offset_in_ECAT_file(mptr,
  			  frame_num, 1, gate_num, data_num, bed_num,
                          segment_num, NULL);

    if (offset_in_file<0)
    { 
      warning("ProjData_to_ECAT7: Error in determining offset into ECAT file for segment %d (f%d, g%d, d%d, b%d)\n"
	      "Maybe the file is too big?\n"
	      "No data written for this segment and all remaining segments",
        segment_num, frame_num, gate_num, data_num, bed_num);
      return Succeeded::no; 
    }

    if (fseek(mptr->fptr, offset_in_file, SEEK_SET))
    {
      warning("\nProjData_to_ECAT7: error in fseek for segment %d (f%d, g%d, d%d, b%d)\n"
            "No data written for this segment and all remaining segments\n",
            segment_num, frame_num, gate_num, data_num, bed_num);
      return Succeeded::no;
    }
    for (int view_num=proj_data.get_min_view_num(); view_num<=proj_data.get_max_view_num(); ++view_num)
      for (int ax_pos_num=proj_data.get_min_axial_pos_num(segment_num); ax_pos_num <= proj_data.get_max_axial_pos_num(segment_num); ++ax_pos_num)
      {
	if (write_data_with_fixed_scale_factor(mptr->fptr, 
					       segment[view_num][ax_pos_num], 
					       output_type_info,
					       scale_factor,
					       output_byte_order,
					       /*can_corrupt_data=*/true)
	    == Succeeded::no)
	  {
	    warning("ProjData_to_ECAT7: error in writing segment %d (f%d, g%d, d%d, b%d)\n"
		    "Not all data written for this segment and none for all remaining segments\n",
		    segment_num, frame_num, gate_num, data_num, bed_num);
	    return Succeeded::no;
	  }
      } // end of loop over ax_pos_num
          
    
  } // end of loop on segments

  cout << endl;
  return Succeeded::yes;
}

} // end of namespace detail

Succeeded 
ProjData_to_ECAT7(MatrixFile *mptr, ProjData const& proj_data, 
                  const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  switch (mptr->mhptr->file_type)
    {
    case AttenCor:
      // always use float to prevent problems with CTI utilities
      return
	detail::ProjData_to_ECAT7_help(mptr, NumericInfo<float>(), proj_data,
			       frame_num, gate_num,data_num, bed_num);
    case Float3dSinogram:
      return 
	detail::ProjData_to_ECAT7_help(mptr, NumericInfo<float>(), proj_data,
			       frame_num, gate_num,data_num, bed_num);
    case Short3dSinogram:
      // Note: this relies on sizeof(short)==2. However, find_ECAT_data_type will find out later if this is not true
      return 
	detail::ProjData_to_ECAT7_help(mptr, NumericInfo<short>(), proj_data,
			       frame_num, gate_num,data_num, bed_num);
    case Byte3dSinogram:
      return 
	detail::ProjData_to_ECAT7_help(mptr, NumericInfo<signed char>(), proj_data,
			       frame_num, gate_num,data_num, bed_num);
    default:
      warning("ProjData_to_ECAT7: unsupported file type %d. No data written.",
	      mptr->mhptr->file_type);
      return Succeeded::no;
    }
  
}

Succeeded 
ProjData_to_ECAT7(ProjData const& proj_data, NumericType output_type,
		  string const & cti_name, string const & orig_name,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num,
                  const bool write_as_attenuation)
{  
  Main_header mhead;
  make_ECAT7_main_header(mhead, orig_name, *proj_data.get_proj_data_info_ptr(),
			   write_as_attenuation, output_type);
  MatrixFile *mptr= matrix_create(cti_name.c_str(), MAT_CREATE, &mhead);   
  Succeeded result =
    ProjData_to_ECAT7(mptr, proj_data, 
                      frame_num, gate_num,data_num, bed_num);
  
  matrix_close(mptr);    
  return result;
}


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif //#ifdef HAVE_LLN_MATRIX
