//
// $Id$
//
/*!
  \file
  \ingroup CTI

  \brief Implementation of routines which convert CTI things into our 
  building blocks and vice versa.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInfo.h"
#include "stir/ProjDataFromStream.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
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

#ifdef STIR_NO_NAMESPACES
// terrible trick to avoid conflict between our Sinogram and matrix::Sinogram
// when we do have namespaces, the conflict can be resolved by using ::Sinogram
#define Sinogram CTISinogram
#endif
#include "matrix.h"
#include "local/stir/CTI/stir_ecat7.h"

#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <memory>


#ifndef STIR_NO_NAMESPACES
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

short find_cti_system_type(const Scanner& scanner)
{
  switch(scanner.get_type())
  {
  case Scanner::E921:
    return 921; 
    
  case Scanner::E931:
    return 931; 
    
  case Scanner::E951:
    return 951; 
    
  case Scanner::E953:
    return 953;

  case Scanner::E961:
    return 961;

  case Scanner::E962:
    return 962; 
    
  case Scanner::E966:
    return 966;

  case Scanner::RPT:
    return 128;
    
  default:
    warning("\nfind_CTI_system_type: scanner \"%s\" currently unsupported. Returning 0.\n", 
      scanner.get_name().c_str());
    return 0;
  }
}

void find_scanner(shared_ptr<Scanner> & scanner_ptr,const Main_header& mhead)
{
  switch(mhead.system_type)
  {
  case 128 : 
    //camera = camRPT; 
    scanner_ptr = new Scanner(Scanner::RPT); 
    printf("Scanner : RPT\n"); 
    break;
  case 931 : 
    //camera = cam931; 
    scanner_ptr = new Scanner(Scanner::E931); 
    printf("Scanner : ECAT 931\n"); break;
  case 951 : 
    //camera = cam951; 
    scanner_ptr = new Scanner(Scanner::E951); 
    printf("Scanner : ECAT 951\n"); break;
  case 953 : 
    //camera = cam953; 
    scanner_ptr = new Scanner(Scanner::E953); 
    printf("Scanner : ECAT 953\n"); 
    break;
  case 921 : 
    //camera = cam953; 
    scanner_ptr = new Scanner(Scanner::E921); 
    printf("Scanner : ECAT 921\n"); 
    break;
  case 961 : 
    //camera = cam953; 
    scanner_ptr = new Scanner(Scanner::E961); 
    printf("Scanner : ECAT 961\n"); 
    break;
  case 966 : 
    //camera = cam953; 
    scanner_ptr = new Scanner(Scanner::E966); 
    printf("Scanner : ECAT 966\n"); 
    break;
  default :  
    
    {	
      // camera = camRPT; 
      scanner_ptr = new Scanner(Scanner::E966); 
      printf("main_header.system_type unknown, defaulting to 966 \n"); 
    }
    break;
  }
}

void find_data_type(NumericType& data_type, ByteOrder& byte_order, const short ecat_data_type)
{
  switch(ecat_data_type)
  {
  case ByteData:
    data_type = NumericType::SCHAR; byte_order = ByteOrder::native; return;
    
  case VAX_Ix2:
    data_type = NumericType("signed integer", 2); byte_order = ByteOrder::little_endian; return;
    
  case SunShort:
    data_type = NumericType("signed integer", 2); byte_order = ByteOrder::big_endian; return;
    
  case VAX_Ix4:
    data_type = NumericType("signed integer", 4); byte_order = ByteOrder::little_endian; return;
    
  case VAX_Rx4:
    data_type = NumericType("float", 4); byte_order = ByteOrder::little_endian; return;
    
  case IeeeFloat:
    data_type =NumericType("float", 4); byte_order = ByteOrder::big_endian; return;
    
  case SunLong:
    data_type = NumericType("signed integer", 4); byte_order = ByteOrder::big_endian; return;
    
  default:
    warning("header.data_type unknown: %d\nAssuming VaxShort\n", ecat_data_type); 
    data_type = NumericType("signed integer", 2); byte_order = ByteOrder::little_endian; return;
  }
}




short find_cti_data_type(const NumericType& type, const ByteOrder& byte_order)
{
  if (!type.signed_type())
    warning("find_cti_data_type: CTI data support only signed types. Using the signed equivalent\n");
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
  warning("find_cti_data_type: CTI does not support data type '%s' of %d bytes.\n",
          number_format.c_str(), size_in_bytes);
  return short(0);
}

/* -------------------------------------------
*	o f f s e t
* -------------------------------------------
*/
static int offset_in_ecat_file (MatrixFile *mptr, int frame, int plane, int gate, int data,
            int bed, int segment, int *plane_size_ptr = NULL)
{
  
  int el_size[15], matnum, strtblk, group = abs(segment),
    plane_size = 0, i, off;
  struct MatDir matdir;
  Scan_subheader scansub;
  Image_subheader imagesub;
  Norm_subheader normsub;
  Attn_subheader attnsub;
  Scan3D_subheader scan3dsub;
  char *prog = "offset_in_ecat_file";
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
          perror("offset_in_ecat_file: error in reading subheader");
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
          perror("offset_in_ecat_file: error in reading subheader");
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
          perror("offset_in_ecat_file: error in reading subheader");
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
          perror("offset_in_ecat_file: error in reading subheader");
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
          perror("offset_in_ecat_file: error in reading subheader");
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
  // first set to default (nonsensical) values
  strcpy(mhead.magic_number, "MATRIX7.0"); // TODO check
  fill_string(mhead.original_file_name, 20);
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
  mhead.bed_elevation= -1.F;
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
  mhead.patient_age = -1.F;
  mhead.patient_height = -1.F;
  mhead.patient_weight = -1.F;
  fill_string(mhead.physician_name,32);
  fill_string(mhead.operator_name,32);
  fill_string(mhead.study_description,32);
  mhead.acquisition_type = 0; 
  mhead.coin_samp_mode = 0; // default to net_trues
  mhead.axial_samp_mode= 0;
  mhead.patient_orientation = -1;
  fill_string(mhead.facility_name, 20);
  mhead.num_planes= 0;
  mhead.num_frames= 1; // used for matnum, so set coherent default values
  mhead.num_gates= 1;
  mhead.num_bed_pos= 0;
  mhead.init_bed_position= -1.F;
  for (int i=0; i<16; i++) mhead.bed_offset[i]= -1.F;
  mhead.plane_separation= -1.F;
  mhead.lwr_sctr_thres= -1;
  mhead.lwr_true_thres= -1;
  mhead.upr_true_thres= -1;  
  fill_string(mhead.user_process_code, 10);
  mhead.acquisition_mode= 0; // default to NORMAL
  mhead.bin_size = -1.F;
  mhead.branching_fraction = -1.F;
  mhead.dose_start_time = 0;
  mhead.dosage = -1.F;
  mhead.well_counter_factor = -1.F;
  fill_string(mhead.data_units,32);
  mhead.septa_state= -1;
  
  // now fill in what we can
  mhead.calibration_factor = 1.F;
  mhead.well_counter_factor=1.F;

  strncpy(mhead.original_file_name, orig_name.c_str(), 20);
  mhead.num_frames= 1; //hdr.num_time_frames;
  
  mhead.system_type= find_cti_system_type(scanner);
  mhead.transaxial_fov= scanner.get_default_num_arccorrected_bins()*scanner.get_default_bin_size()/10;
  mhead.intrinsic_tilt = scanner.get_default_intrinsic_tilt();
  mhead.bin_size = scanner.get_default_bin_size();
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


// mhead.file_type is set to Float3dSinogram. WARNING maybe you have to adjust file_type
void make_ECAT7_main_header(Main_header& mhead,
			    const string& orig_name,
                            ProjDataInfo const & proj_data_info
                            )
{
  
  make_ECAT7_main_header(mhead, *proj_data_info.get_scanner_ptr(), orig_name);
  
  // extra main parameters that depend on data type
  mhead.file_type= Float3dSinogram;
  
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
void scan_subheader_zero_fill_aux(Subheader& shead) 
{ 
  shead.data_type= -1;
  shead.num_dimensions = -1;
  shead.num_r_elements = -1;
  shead.num_angles = -1;
  shead.ring_difference = -1;
  shead.storage_order = -1;
  shead.x_resolution = -1.F;
  shead.v_resolution = -1.F;
  shead.z_resolution = -1.F;
  shead.w_resolution = -1.F;
  shead.scale_factor= -1.F;
}

void scan_subheader_zero_fill(Scan3D_subheader& shead) 
{ 
  shead.corrections_applied = -1;
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
  for(int i=0;i<16;++i) shead.uncor_singles[i] = -1.F;

}

void scan_subheader_zero_fill(Attn_subheader& shead) 
{
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



//! A utility function only called by make_subheader_for_ecat7(..., ProjDataInfo&)
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
make_subheader_for_ecat7_aux(SUBHEADERPTR sub_header_ptr, 
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
make_subheader_for_ecat7(Attn_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         )
{
  make_subheader_for_ecat7_aux(&shead, shead.z_elements, shead.span,
                     mhead, proj_data_info);
  if (dynamic_cast<ProjDataInfoCylindricalNoArcCorr const * const>(&proj_data_info))
  {
    warning("make_subheader_for_ecat7: data is not arc-corrected but info is not available in CTI attenuation subheader\n");
  }
}
    
// WARNING data_type has to be set
void
make_subheader_for_ecat7(Scan3D_subheader& shead, 
                         const Main_header& mhead,
                         const ProjDataInfo& proj_data_info
                         )
{
  make_subheader_for_ecat7_aux(&shead, shead.num_z_elements, shead.axial_compression,
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
    warning("make_subheader_for_ecat7: unknown type of proj_data_info. Setting data to arc-corrected anyway\n");
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
  find_data_type(data_type, byte_order, sub_header_ptr->data_type);

  if (bin_size != scanner_ptr->get_default_bin_size())
  {
    warning("Bin size from header (%g) does not agree with expected value %g\nfor scanner %s. Using expected value...\n",
	    bin_size, 
	    scanner_ptr->get_default_bin_size(), 
	    scanner_ptr->get_name().c_str());
    // TODO
    //scanner_ptr->set_bin_size(bin_size);
  }
  // TODO more checks on FOV etc.
  
  shared_ptr<ProjDataInfo> pdi_ptr =
    ProjDataInfo::ProjDataInfoCTI(scanner_ptr, span, max_delta, 
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
    offset_in_ecat_file(mptr,
			matval.frame, 1, matval.gate, matval.data, matval.bed,
			0, NULL);
  // KT 14/05/2002 added error check
  if (offset_in_ecat_file<0)
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

Succeeded 
write_basic_interfile_header_for_ecat7(const string& ecat7_filename,
                                       int frame, int gate, int data,
                                       int bed)
{
  

  
  MatrixFile * const mptr = matrix_open( ecat7_filename.c_str(), MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror( ecat7_filename.c_str());
    return Succeeded::no;
  }
  if (mptr->mhptr->sw_version < V7)
  { 
    warning("This seems to be an ECAT 6 file. I'm not writing any header...\n"); 
    return Succeeded::no; 
  }
  

  const int matnum = mat_numcod (frame, 1, gate, data, bed);
  MatrixData* matrix = matrix_read( mptr, matnum, MAT_SUB_HEADER);
  
  if (matrix==NULL)
  { 
    warning("Matrix not found at \"%d,1,%d,%d,%d\". I'm not writing any header...\n",
       frame, 1, gate, data, bed);
    return Succeeded::no;
  }
  
  char *header_filename = new char[ecat7_filename.size() + 100];
  {
    strcpy(header_filename, ecat7_filename.c_str());
    // keep extension, just in case we would have conflicts otherwise
    // but replace the . with a _
    char * dot_ptr = strchr(find_filename(header_filename),'.');
    if (dot_ptr != NULL)
      *dot_ptr = '_';
    // now add stuff to say which frame, gate, bed, data this was
    sprintf(header_filename+strlen(header_filename), "_f%dg%db%dd%d", 
	    frame, gate, bed, data);
  }
  
  switch (mptr->mhptr->file_type)
  {
  //case PetImage: // TODO this probably has subheaders?
  case ByteVolume:
  case PetVolume:
    {
      Image_subheader *sub_header_ptr=
        reinterpret_cast<Image_subheader*>(matrix->shptr);
      
      if(sub_header_ptr->num_dimensions != 3)
        warning("Expected subheader_ptr->num_dimensions==3\n");
      const Coordinate3D<int> dimensions(matrix->zdim,
        matrix->ydim,
        matrix->xdim);
      const CartesianCoordinate3D<float> voxel_size(matrix->z_size * 10,
        matrix->y_size * 10,
        matrix->pixel_size * 10); // convert to mm
      /* not used yet
      const Coordinate3D<float> origin(matrix->z_origin * 10,
        matrix->y_origin * 10,
        matrix->x_origin * 10); // convert to mm
      */
      const float scale_factor = matrix->scale_factor;
      
      NumericType data_type;
      ByteOrder byte_order;
      find_data_type(data_type, byte_order, matrix->data_type);
      
      const long offset_in_file =
        offset_in_ecat_file(mptr, frame, 1, gate, data, bed, 0, NULL);
      // KT 14/05/2002 added error check
      if (offset_in_ecat_file<0)
      { 
        warning("Error in determining offset into ECAT file. I'm not writing any header...\n"); 
        return Succeeded::no; 
      }     
      
      VectorWithOffset<float> scaling_factors(1);
      VectorWithOffset<unsigned long> file_offsets(1);
      scaling_factors[0] = scale_factor;
      file_offsets[0] = static_cast<unsigned long>(offset_in_file);
      strcat(header_filename, ".hv");
      write_basic_interfile_image_header(header_filename, ecat7_filename,
        dimensions, voxel_size, data_type,byte_order,
        scaling_factors,
        file_offsets);
      break;
    }
          
  case AttenCor:   		
  case Byte3dSinogram:
  case Short3dSinogram:
  case Float3dSinogram :
    {
      shared_ptr<iostream> stream_ptr = 
	new fstream(ecat7_filename.c_str(), ios::in | ios::binary);
      
      ProjDataFromStream* pdfs_ptr = 
	make_pdfs_from_matrix(mptr, matrix, stream_ptr);
     
      if (pdfs_ptr == NULL)
	return Succeeded::no;
      
      strcat(header_filename, ".hs");
      write_basic_interfile_PDFS_header(header_filename, ecat7_filename, *pdfs_ptr);
      delete pdfs_ptr;

      break;
    }
    
        
  default:
    warning("File type not handled. I'm not writing any Interfile headers...\n");
    return Succeeded::no;
  }
  
  delete header_filename;
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
    scanner_ptr->get_default_num_arccorrected_bins()/128;

  ihead.decay_corr_fctr= 1;

#if 0  
  // attempt to write this ourselves, but we'd need to write the subheader, and we 
  // don't have neat functions for written Arrays to FILE anyway (only to streams)
  NumericType data_type;
  ByteOrder byte_order;
  find_data_type(data_type, byte_order, mhead.data_type);
  
  const long offset_in_file =
    offset_in_ecat_file(mptr, frame, 1, gate, data, bed, 0, NULL);
  
  // KT 14/05/2002 added error check
  if (offset_in_ecat_file<0)
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
ProjData_to_ECAT7(MatrixFile *mptr, ProjData const& proj_data, 
                  const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  const Main_header& mhead = *(mptr->mhptr);
  if (mhead.file_type!= Float3dSinogram && mhead.file_type != AttenCor)
  {
    warning("ProjData_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.file_type should be Float3dSinogram or AttenCor\n",
            frame_num, gate_num, data_num, bed_num);
    return Succeeded::no;
  }
  {
    int num_planes = 0;
    for(int segment_num=proj_data.get_min_segment_num();
        segment_num <= proj_data.get_max_segment_num();
        ++segment_num)
     num_planes+= proj_data.get_num_axial_poss(segment_num);
  
    if (mhead.num_planes!=num_planes) // TODO check if this is the usual convention
    {
      warning("ProjData_to_ECAT7: converting (f%d, g%d, d%d, b%d)\n"
              "Main header.num_planes should be %d\n",
              frame_num, gate_num, data_num, bed_num,num_planes);
      return Succeeded::no;
    }
  }
  
  Scan3D_subheader scan3d_shead; 
  Attn_subheader attn_shead;

  if (mhead.file_type == AttenCor)
  {
    make_subheader_for_ecat7(attn_shead, mhead, *proj_data.get_proj_data_info_ptr());
    // Setup remaining subheader params
    attn_shead.data_type= IeeeFloat;
    attn_shead.scale_factor= 1.F; 
    attn_shead.storage_order = ElAxVwRd;
  }
  else
  {
    make_subheader_for_ecat7(scan3d_shead, mhead, *proj_data.get_proj_data_info_ptr());
    // Setup remaining subheader params
    scan3d_shead.data_type= IeeeFloat;
    scan3d_shead.loss_correction_fctr= 1.F; 
    scan3d_shead.scale_factor= 1.F; 
    scan3d_shead.storage_order = ElAxVwRd;
  }

  // allocate space in file, and write subheader
  {
    const int ERROR=-1;
    const int plane_size= proj_data.get_num_tangential_poss() * proj_data.get_num_views();
    
    // TODO relies on writing floats
    // TODO only ok if main_header.num_planes is set as above
    int nblks = (mhead.num_planes*plane_size*sizeof(float)+511)/512;
    
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
  
  NumericType data_type;
  ByteOrder byte_order;
  find_data_type(data_type, byte_order, 
                 mhead.file_type == AttenCor?attn_shead.data_type:scan3d_shead.data_type);

  cout<<endl<<"Processing segment number:";
  
  for(int segment_num=proj_data.get_min_segment_num();
      segment_num <= proj_data.get_max_segment_num();
      ++segment_num)
  {    
    cout<<"  "<<segment_num;

    // read the segment
    // note: can't be a const variable as we'll potentially byteswap it
    SegmentByView<float> segment = 
      proj_data.get_segment_by_view(segment_num);

    const long offset_in_file =
      offset_in_ecat_file(mptr,
  			  frame_num, 1, gate_num, data_num, bed_num,
                          segment_num, NULL);
    // KT 14/05/2002 added error check
    if (offset_in_ecat_file<0)
    { 
      warning("Error in determining offset into ECAT file for segment %d (f%d, g%d, d%d, b%d)\n"
        "No data written for this segment and all remaining segments\n",
        segment_num, frame_num, gate_num, data_num, bed_num);
      return Succeeded::no; 
    }

    if (fseek(mptr->fptr, offset_in_file, SEEK_SET))
    {
      warning("ProjData_to_ECAT7: error in fseek for segment %d (f%d, g%d, d%d, b%d)\n"
            "No data written for this segment and all remaining segments\n",
            segment_num, frame_num, gate_num, data_num, bed_num);
      return Succeeded::no;
    }
    for (int view_num=proj_data.get_min_view_num(); view_num<=proj_data.get_max_view_num(); ++view_num)
      for (int ax_pos_num=proj_data.get_min_axial_pos_num(segment_num); ax_pos_num <= proj_data.get_max_axial_pos_num(segment_num); ++ax_pos_num)
      {
        Array<1,float>& array = segment[view_num][ax_pos_num];

        // note for byte swapping
        // if necessary, we swap, then write. We really should swap back, but as the
        // data isn't needed after this, we don't.
        // Note that swapping back could be less expensive than copying the data 
        // somewhere and do the swap there
        const size_t size_to_write = 
          sizeof(*array.begin()) * static_cast<size_t>(array.get_length());
        if (size_to_write==0)
          continue;
        if (!byte_order.is_native_order())
          for(int i=array.get_min_index(); i<=array.get_max_index(); ++i)
            ByteOrder::swap_order(array[i]);
        const size_t size_written = 
          fwrite(reinterpret_cast<const char *>(array.get_data_ptr()), 
                 1, size_to_write, mptr->fptr);
        array.release_data_ptr();
        if (size_written != size_to_write)
        {
          warning("ProjData_to_ECAT7: error in writing for segment %d (f%d, g%d, d%d, b%d)\n"
            "Not all data written for this segment and none for all remaining segments\n",
            segment_num, frame_num, gate_num, data_num, bed_num);
          return Succeeded::no;
        }
      } // end of loop over ax_pos_num
          
    
  } // end of loop on segments

  return Succeeded::yes;
}


Succeeded 
ProjData_to_ECAT7(ProjData const& proj_data, string const & cti_name, string const & orig_name,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num,
                  const bool write_as_attenuation)
{  
  Main_header mhead;
  make_ECAT7_main_header(mhead, orig_name, *proj_data.get_proj_data_info_ptr());
  if (write_as_attenuation)
    mhead.file_type = AttenCor;
  else
    mhead.file_type = Float3dSinogram;
  
  MatrixFile *mptr= matrix_create(cti_name.c_str(), MAT_CREATE, &mhead);   
  Succeeded result =
    ProjData_to_ECAT7(mptr, proj_data, 
                      frame_num, gate_num,data_num, bed_num);
  
  matrix_close(mptr);    
  return result;
}

END_NAMESPACE_ECAT7

END_NAMESPACE_STIR
