//
// $Id$
//

/*! 
\file
\brief Utility to make Interfile headers for ECAT7 data
\author Kris Thielemans
\author Cristina de Oliveira (offset_in_ecat_file function)
\author PARAPET project
$Date$
$Revision$

\warning This only works with some CTI file_types. In particular, it does NOT
work with the ECAT6 files_types, as then there are subheaders 'in' the 
datasets.

\warning Implementation uses the Louvain la Neuve Ecat library. So, it will
only work on systems where this library works properly.

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
/* 
  History
  KT first version
  KT 25/10/2000 swapped segment order
*/

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <stdarg.h>

#include "stir/ProjDataInfo.h"
#include "stir/ProjDataFromStream.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/CartesianCoordinate3D.h"
#include <iostream>
#include <fstream>
#include <string>

#ifdef STIR_NO_NAMESPACES
// terrible trick to avoid conflict between our Sinogram and matrix::Sinogram
// when we do have namespaces, the conflict can be resolved by using ::Sinogram
#define Sinogram CTISinogram
#endif

#include "matrix.h"

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::ios;
using std::iostream;
using std::fstream;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

/* ------------------------------------
 *	print_debug
 * ------------------------------------*/
int print_debug (char const * const fname, char *format, ...) 
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


void find_scanner(shared_ptr<Scanner> & scanner_ptr,const Main_header& mhead)
{
  switch(mhead.system_type)
  {
  case 42:
    scanner_ptr =  new Scanner(Scanner::RATPET);
    printf("Scanner: RATPET\n");
    break;
  case 128 : 
    scanner_ptr = new Scanner(Scanner::RPT); 
    printf("Scanner : RPT\n"); 
    break;
  case 931 : 
    scanner_ptr = new Scanner(Scanner::E931); 
    printf("Scanner : ECAT 931\n"); break;
  case 951 : 
    scanner_ptr = new Scanner(Scanner::E951); 
    printf("Scanner : ECAT 951\n"); break;
  case 953 : 
    scanner_ptr = new Scanner(Scanner::E953); 
    printf("Scanner : ECAT 953\n"); 
    break;
  case 921 : 
    scanner_ptr = new Scanner(Scanner::E921); 
    printf("Scanner : ECAT 921\n"); 
    break;
  case 925 : 
    scanner_ptr = new Scanner(Scanner::E925); 
    printf("Scanner : ECAT 925\n"); 
    break;
  case 961 : 
    scanner_ptr = new Scanner(Scanner::E961); 
    printf("Scanner : ECAT 961\n"); 
    break;
  case 962 : 
    scanner_ptr = new Scanner(Scanner::E962); 
    printf("Scanner : ECAT 962\n"); 
    break;
  case 966 : 
    scanner_ptr = new Scanner(Scanner::E966); 
    printf("Scanner : ECAT 966\n"); 
    break;
  default :      
    {	
      scanner_ptr = new Scanner(Scanner::E962); 
      printf("main_header.system_type unknown, defaulting to 962 (ECAT HR+)\n"); 
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
    return;
  }
}

/* -------------------------------------------
*	o f f s e t
* -------------------------------------------
*/
int offset_in_ecat_file (MatrixFile *mptr, int frame, int plane, int gate, int data,
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
      mat_read_scan_subheader (mptr->fptr, mptr->mhptr, strtblk, &scansub);
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
      
      mat_read_image_subheader (mptr->fptr, mptr->mhptr, strtblk, &imagesub);
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
        mat_read_attn_subheader (mptr->fptr, mptr->mhptr, strtblk, &attnsub);
        
        
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
      mat_read_norm_subheader (mptr->fptr, mptr->mhptr, strtblk, &normsub);
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
      
      mat_read_Scan3D_subheader (mptr->fptr, mptr->mhptr, strtblk, &scan3dsub);
      
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
  case PetImage:
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
    warning("File type not handled. I'm not writing any headers...\n");
    return Succeeded::no;
  }
  
  delete header_filename;
  return Succeeded::yes;
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int	
main( int argc, char **argv)
{
  MatrixFile *mptr;
  
  if (argc<2)
  {
    cerr << "usage    : "<< argv[0] << " filename\n";
    exit(EXIT_FAILURE);
  }

  mptr = matrix_open( argv[1], MAT_READ_ONLY, MAT_UNKNOWN_FTYPE);
  if (!mptr) {
    matrix_perror(argv[1]);
    exit(EXIT_FAILURE);
  }
  
  const int num_frames = std::max(static_cast<int>( mptr->mhptr->num_frames),1);
  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_bed_poss = static_cast<int>( mptr->mhptr->num_bed_pos) + 1;
  const int num_gates = std::max(static_cast<int>( mptr->mhptr->num_gates),1);

  fclose(mptr->fptr);
  delete mptr;

  if (ask("Attempt all data-sets (Y) or single data-set (N)", true))
  {
    const int data_num=ask_num("Data number ? ",0,8, 0);

    for (int frame_num=1; frame_num<=num_frames;++frame_num)
      for (int bed_num=0; bed_num<num_bed_poss;++bed_num)
        for (int gate_num=1; gate_num<=num_gates;++gate_num)
          write_basic_interfile_header_for_ecat7( argv[1], 
						  frame_num, gate_num, data_num, bed_num);
  }
  else
  {
    const int frame_num=ask_num("Frame number ? ",1,num_frames, 1);
    const int bed_num=ask_num("Bed number ? ",0,num_bed_poss-1, 0);
    const int gate_num=ask_num("Gate number ? ",1,num_gates, 1);
    const int data_num=ask_num("Data number ? ",0,8, 0);
    /*
    const int plane_num=ask_num("Plane num ?",1,100,1);
    const int segment_num=ask_num("Segment num ?",-100,100,1);
    
     std::cerr <<
     offset_in_ecat_file (mptr, frame_num, plane_num, gate_num, data_num,
     bed_num,  segment_num, NULL) << std::endl;
    */
    write_basic_interfile_header_for_ecat7( argv[1], frame_num, gate_num, data_num,
      bed_num);
  }
  return EXIT_SUCCESS;
}
