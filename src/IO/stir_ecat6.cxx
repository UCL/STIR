//
// $Id$
//
/*!
  \file
  \ingroup ECAT

  \brief Implementation of routines which convert CTI things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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


#include "stir/IO/interfile.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfo.h"
#include "stir/IndexRange3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ByteOrder.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/convert_array.h"
#include "stir/IndexRange2D.h"
#include "stir/utilities.h"

#include "stir/Scanner.h" 
#include "stir/IO/ecat6_utils.h"
#include "stir/IO/stir_ecat6.h"
#ifndef STIR_ORIGINAL_ECAT6
#include "stir/IO/stir_ecat7.h"
#endif

#include "boost/cstdint.hpp" 
#include "boost/static_assert.hpp" 
#include "boost/scoped_array.hpp" 
#include <iostream>
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::min;
#endif

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6

static void cti_data_to_float_Array(Array<2,float>&out, 
                             char const * const buffer, const float scale_factor, int dtype);


/*
  \brief reads data from a CTI file into a Sinogram object
  \param buffer is supposed to be a pre-allocated buffer (which will be modified)

  It applies all scale factors.
  \warning \a buffer has to be allocated with a size at least as large as the
  multiple of MatBLKSIZE that fits the whole sinogram (because it uses cti_wblk).
  */
static void read_sinogram(Sinogram<float>& sino_2D,
                   char *buffer, 
		   FILE*fptr, 
		   int mat_index, 
		   int frame, int gate, int data, int bed);

static bool is_ECAT6_file(ECAT6_Main_header& mhead, const string& filename)
{
  //check if it's ECAT 6 
  FILE * cti_fptr=fopen(filename.c_str(), "rb"); 

  if(!cti_fptr ||
     cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) 
    {
      // this is funny as it's just reading a bunch of bytes. anyway. we'll assume it isn't ECAT6
      return false;
    }
  else
    {
      fclose(cti_fptr);
      // do some checks on the main header
      return 
	mhead.sw_version>=0 && mhead.sw_version<=69  &&
	( mhead.file_type == matScanFile ||
	  mhead.file_type == matImageFile ||
	  mhead.file_type == matAttenFile ||
	  mhead.file_type == matNormFile) &&
	mhead.num_frames>0;
    }
}

  
bool is_ECAT6_file(const string& filename)
{
  ECAT6_Main_header mhead;
  return is_ECAT6_file(mhead, filename);
}

bool is_ECAT6_image_file(const string& filename)
{
  ECAT6_Main_header mhead;
  return is_ECAT6_file(mhead, filename) &&
    mhead.file_type ==matImageFile;
}


bool is_ECAT6_emission_file(const string& filename)
{
  ECAT6_Main_header mhead;
  return is_ECAT6_file(mhead, filename) &&
    mhead.file_type ==matScanFile;
}


bool is_ECAT6_attenuation_file(const string& filename)
{
  ECAT6_Main_header mhead;
  return is_ECAT6_file(mhead, filename) &&
    mhead.file_type ==matAttenFile;
}



Scanner* find_scanner_from_ECAT6_Main_header(const ECAT6_Main_header& mhead)
{
  // we could do more effort here by checking some values of other fields than system_type.
  // TODO

  
  Scanner * scanner_ptr = 
    find_scanner_from_ECAT_system_type(mhead.system_type);
  return scanner_ptr;
}

void make_ECAT6_Main_header(ECAT6_Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name                     
                            )
{
#ifndef STIR_ORIGINAL_ECAT6
  ecat::ecat7::make_ECAT7_main_header(mhead, scanner, orig_name);
  strcpy(mhead.magic_number, "MATRIX6.4");
  mhead.sw_version= 64;
#else
  mhead= main_zero_fill();
  mhead.calibration_factor = 1.F;
  
  // other header parameters
  
  strncpy(mhead.original_file_name, orig_name.c_str(), 20);
  mhead.num_frames= 1; //hdr.num_time_frames;
  // cti_utils routines always write data as VAX short
  mhead.data_type= ECAT_I2_little_endian_data_type;
  
  mhead.system_type= find_ECAT_system_type(scanner);
  mhead.axial_fov= scanner.get_num_rings()*scanner.get_ring_spacing()/10;
  mhead.transaxial_fov= scanner.get_default_num_arccorrected_bins()*scanner.get_default_bin_size()/10;
  
  mhead.plane_separation= scanner.get_ring_spacing()/2/10;
  //WRONG mhead.gantry_tilt= scanner.get_default_intrinsic_tilt();
#endif // STIR_ORIGINAL_ECAT6
}

void make_ECAT6_Main_header(ECAT6_Main_header& mhead,
			    Scanner const& scanner,
                            const string& orig_name,
                            DiscretisedDensity<3,float> const & density
                            )
{
  make_ECAT6_Main_header(mhead, scanner, orig_name);
  
  DiscretisedDensityOnCartesianGrid<3,float> const & image =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(density);

  
  // extra main parameters that depend on data type
  mhead.file_type= matImageFile;
  mhead.num_planes=image.get_length();
  mhead.plane_separation=image.get_grid_spacing()[1]/10; // convert to cm
}

void make_ECAT6_Main_header(ECAT6_Main_header& mhead,
			    const string& orig_name,
                            ProjDataInfo const & proj_data_info
                            )
{
  
  make_ECAT6_Main_header(mhead, *proj_data_info.get_scanner_ptr(), orig_name);
  
  // extra main parameters that depend on data type
  mhead.file_type= matScanFile;
  
  mhead.num_planes = 0;
  for(int segment_num=proj_data_info.get_min_segment_num();
      segment_num <= proj_data_info.get_max_segment_num();
      ++segment_num)
    mhead.num_planes+= proj_data_info.get_num_axial_poss(segment_num);
  
  mhead.plane_separation=proj_data_info.get_scanner_ptr()->get_ring_spacing()/10/2;
}

VoxelsOnCartesianGrid<float> *
ECAT6_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      FILE *cti_fptr, const ECAT6_Main_header & mhead)
{
  MatDir entry;
  Image_subheader ihead;


  VoxelsOnCartesianGrid<float> * image_ptr = 0;
  
  // read first subheader to find dimensions
  {
    long matnum = cti_numcod(frame_num, 1,gate_num, data_num, bed_num);
    
    if(!cti_lookup(cti_fptr, &mhead,  matnum, &entry)) { // get entry
      error("\nCouldn't find matnum %d in specified file.\n", matnum);            
    }
    if(cti_read_image_subheader(cti_fptr, &mhead, entry.strtblk, &ihead)!=EXIT_SUCCESS)
      { 
	error("\nUnable to look up image subheader\n");
      }
  }
  
#ifndef STIR_ORIGINAL_ECAT6
  const int x_size = ihead.x_dimension;
  const int y_size = ihead.y_dimension;
  const int z_size = ihead.z_dimension;
#else
  const int x_size = ihead.dimension_1;
  const int y_size = ihead.dimension_2;
  const int z_size = mhead.num_planes;
#endif
  const int min_z = 0; 
  
  IndexRange3D range_3D (0,z_size-1,
			 -y_size/2,(-y_size/2)+y_size-1,
			 -x_size/2,(-x_size/2)+x_size-1);
  
#ifndef STIR_ORIGINAL_ECAT6
  CartesianCoordinate3D<float> 
    voxel_size(ihead.z_pixel_size*10,ihead.y_pixel_size*10,ihead.x_pixel_size*10);
  CartesianCoordinate3D<float> 
    origin(ihead.z_offset, ihead.y_offset*10, ihead.x_offset*10);
#else
  CartesianCoordinate3D<float> 
    voxel_size(ihead.slice_width*10,ihead.pixel_size*10,ihead.pixel_size*10);
  CartesianCoordinate3D<float> 
    origin(0, ihead.y_origin*10, ihead.x_origin*10);
#endif
  
  
  image_ptr = new VoxelsOnCartesianGrid<float> (range_3D, origin, voxel_size);
  
  NumericType type;
  ByteOrder byte_order;
  find_type_from_ECAT_data_type(type, byte_order, ihead.data_type);
  // allocation for buffer. Provide enough space for a multiple of MatBLKSIZE  
  const size_t cti_data_size = x_size*y_size*type.size_in_bytes()+ MatBLKSIZE;
  char * cti_data= new char[cti_data_size];

  for(int z=0; z<mhead.num_planes; z++) 
    { // loop upon planes
      
      long matnum = cti_numcod(frame_num, z+1,gate_num, data_num, bed_num);
        
      if(!cti_lookup(cti_fptr, &mhead, matnum, &entry)) { // get entry
	error("\nCouldn't find matnum %d in specified file.\n", matnum);     
      }

      if(cti_read_image_subheader(cti_fptr, &mhead, entry.strtblk, &ihead)==EXIT_FAILURE) { // get ihead for plane z
	error("\nUnable to look up image subheader\n");
      }
        
#ifndef STIR_ORIGINAL_ECAT6
      const CartesianCoordinate3D<float> sub_head_origin(ihead.z_offset, ihead.y_offset*10, ihead.x_offset*10);        
#else // STIR_ORIGINAL_ECAT6
      const CartesianCoordinate3D<float> sub_head_origin(0, ihead.y_origin*10, ihead.x_origin*10);
#endif // STIR_ORIGINAL_ECAT6
      {
	if (norm(image_ptr->get_origin() - sub_head_origin) > .01F)
          {
		    warning("ECAT6_to_VoxelsOnCartesianGrid: x,y offset in subheader of plane %d does not agree with plane 1. Ignoring it...\n",
		    z+1);
          }
      }

#ifndef STIR_ORIGINAL_ECAT6
      float scale_factor = ihead.scale_factor; 
#else // STIR_ORIGINAL_ECAT6
      float scale_factor = ihead.quant_scale;
#endif
      if(cti_rblk (cti_fptr, entry.strtblk+1, cti_data, entry.endblk-entry.strtblk)!=EXIT_SUCCESS) { // get data
	error("\nUnable to read data\n");            
      }
      if (file_data_to_host(cti_data, entry.endblk-entry.strtblk, ihead.data_type)!=EXIT_SUCCESS)
        error("\nerror converting to host data format\n");
      cti_data_to_float_Array((*image_ptr)[z+min_z],cti_data, scale_factor, ihead.data_type);
#if 0
      NumericType type;
      ByteOrder byte_order;
      find_type_from_cti_data_type(type, byte_order, ihead.data_type);
        
      for(int y=0; y<y_size; y++)
	for(int x=0; x<x_size; x++)
          {
            (*image_ptr)[z+min_z][y+min_y][x+min_x]=scale_factor*cti_data[y*x_size+x];
          }
#endif
    } // end loop on planes
    
  delete[] cti_data;
  return image_ptr;
}

void ECAT6_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
		   int max_ring_diff, bool arccorrected,
                   const string& data_name, FILE *cti_fptr, const ECAT6_Main_header &mhead)
{
  shared_ptr<Scanner> scanner_ptr = find_scanner_from_ECAT6_Main_header(mhead);
  cout << "Scanner determined from ECAT6_Main_header: " << scanner_ptr->get_name() << endl;
  if (scanner_ptr->get_type() == Scanner::Unknown_scanner ||
      scanner_ptr->get_type() == Scanner::User_defined_scanner)
  {
    warning("ECAT6_to_PDFS: Couldn't determine the scanner \n"
      "(Main_header.system_type=%d), defaulting to 953.\n"
      "This will give dramatic problems when the number of rings of your scanner is NOT 16.\n",
	    mhead.system_type); 
    scanner_ptr = new Scanner(Scanner::E953);
  }

  
  const int num_rings = scanner_ptr->get_num_rings();

  // ECAT 6 does not have a flag for 3D vs. 2D, so we guess it first from num_planes
  bool is_3D_file = (mhead.num_planes >  2*num_rings-1);
  if (!is_3D_file)
  {
    // better make sure by checking if plane (5,5) is not in its '3D' place
    MatDir entry;
    const int mat_index= cti_rings2plane(num_rings, 5,5); 
    const long matnum= cti_numcod(frame_num, mat_index,gate_num, data_num, bed_num);
    // KT 18/08/2000 add !=0 to prevent compiler warning on conversion from int to bool
    is_3D_file = cti_lookup (cti_fptr, &mhead, matnum, &entry)!=0;
  }
  int span = 1;

  if (!is_3D_file)
  {
    warning("I'm guessing this is a stack of 2D sinograms\n");
    if(mhead.num_planes == 2*num_rings-1)
    {
      span=3; 
      max_ring_diff = 1;
    }
    else if (mhead.num_planes == num_rings)
    {
      span=1;
      max_ring_diff = 0;
    }
    else
    {
      error("Impossible num_planes: %d\n", mhead.num_planes);
    }
  }
  else
  {
    if (max_ring_diff < 0) 
      max_ring_diff = num_rings-1;
    const int num_sinos = 
      (2*max_ring_diff+1) * num_rings  - (max_ring_diff+1)*max_ring_diff;
    
    if (num_sinos > mhead.num_planes)
    warning("\n\aWarning: header says not enough planes in the file: %d (expected %d)."
	    "Continuing anyway...", mhead.num_planes, num_sinos);
  }
	
  // construct a ProjDataFromStream object
  shared_ptr<ProjDataFromStream>  proj_data =  NULL;
  ScanInfoRec scanParams;

  { 
    // read first subheader for dimensions
    {     
      // use temporary copy to avoid overwriting mhead argument
      ECAT6_Main_header mhead_copy;

      long matnum= cti_numcod(frame_num, 1,gate_num, data_num, bed_num);
      switch(mhead.file_type)
      {
      case matScanFile:
        {
          Scan_subheader shead;
          if (get_scanheaders(cti_fptr, matnum, &mhead_copy, &shead, &scanParams)!= EXIT_SUCCESS)
            error("Error reading matnum %d\n", matnum);
          break;
        }   
      case matAttenFile:
        {
          Attn_subheader shead;        
          if(get_attnheaders (cti_fptr, matnum, &mhead_copy, 
			      &shead, &scanParams)!= EXIT_SUCCESS)
            error("Error reading matnum %d\n", matnum);
          break;
        }
      case matNormFile:
        {
          Norm_subheader shead;        
          if(get_normheaders (cti_fptr, matnum, &mhead_copy, 
			      &shead, &scanParams)!= EXIT_SUCCESS)
            error("Error reading matnum %d\n", matnum);
          break;
        }
      
      default:
        error("ECAT6_to_PDFS: unsupported file type %d\n",mhead.file_type);
      }
    }
    const int num_views = scanParams.nviews;
    const int num_tangential_poss = scanParams.nprojs; 
    
    
    shared_ptr<ProjDataInfo> p_data_info= 
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,span,max_ring_diff,num_views,num_tangential_poss,arccorrected); 
    
    
    ProjDataFromStream::StorageOrder  storage_order=
      ProjDataFromStream::Segment_AxialPos_View_TangPos;
    
    // TODO replace these char* things with string based extension stuff
    boost::scoped_array<char> actual_data_name(new char[data_name.size() + 4]);
    strcpy(actual_data_name.get(), data_name.c_str());
    // KT 30/05/2002 make sure that a filename ending on .hs is treated correctly
    {
      const char * const extension = strchr(find_filename(actual_data_name.get()),'.');
      if (extension!=NULL && strcmp(extension, ".hs")==0)
        replace_extension(actual_data_name.get(), ".s");
      else
        add_extension(actual_data_name.get(), ".s");
    }
    shared_ptr<iostream> sino_stream =
      new fstream (actual_data_name.get(), ios::out| ios::binary);
    
    if (!sino_stream->good())
    {
      error("ECAT6cti_to_PDFS: error opening file %s\n",actual_data_name.get());
    }
    
    
    proj_data = 
      new ProjDataFromStream(p_data_info,sino_stream, streamoff(0), storage_order);
    
    write_basic_interfile_PDFS_header(actual_data_name.get(), *proj_data);
  }


  // write to proj_data
  {
    NumericType type;
    ByteOrder byte_order;
    find_type_from_ECAT_data_type(type, byte_order, scanParams.data_type);
    // allocation for buffer. Provide enough space for a multiple of MatBLKSIZE  
    const size_t cti_data_size = 
      proj_data->get_num_tangential_poss()*proj_data->get_num_views()*type.size_in_bytes()+ MatBLKSIZE;
    //use scoped_array to auto-delete the memory
    boost::scoped_array<char> cti_data_sptr(new char[cti_data_size]);
    char * cti_data= cti_data_sptr.get();
     
    cout<<"\nProcessing segment number:";
    
    if (is_3D_file)
    {
      for(int w=0; w<=max_ring_diff; w++) 
      { // loop on segment number
        
        // positive ring difference
        cout<<"  "<<w;
        int num_axial_poss= num_rings-w;
        
        for(int ring1=0; ring1<num_axial_poss; ring1++) { // ring order: 0-0,1-1,..,15-15 then 0-1,1-2,..,14-15
          int ring2=ring1+w; // ring1<=ring2
          int mat_index= cti_rings2plane(num_rings, ring1, ring2);          
          Sinogram<float> sino_2D = proj_data->get_empty_sinogram(ring1,w);
          //TODO remove as will be set below
	  proj_data->set_sinogram(sino_2D);  
          read_sinogram(sino_2D, cti_data, cti_fptr, mat_index, 
            frame_num, gate_num, data_num, bed_num);
          proj_data->set_sinogram(sino_2D);                     
        }
        
        // negative ring difference
        if(w>0) {
          cout<<"  "<<-w;
          for(int ring2=0; ring2<num_axial_poss; ring2++) { // ring order: 0-1,2-1,..,15-14 then 2-0,3-1,..,15-13
            int ring1=ring2+w; // ring1>ring2
            int mat_index= cti_rings2plane(num_rings, ring1, ring2);
            Sinogram<float> sino_2D = proj_data->get_empty_sinogram(ring2,-w,false);
            read_sinogram(sino_2D, cti_data,
              cti_fptr, mat_index, 
              frame_num, gate_num, data_num, bed_num);
            
            proj_data->set_sinogram(sino_2D);           
          }
        }
      } // end of loop on segment number
    } // end of 3D case
    else
    {
      // 2D case
      cout << "0\n";
      for(int z=0; z<proj_data->get_num_axial_poss(0); z++)
      {
        Sinogram<float> sino_2D = proj_data->get_empty_sinogram(z,0,false);
        read_sinogram(sino_2D, cti_data, cti_fptr, z+1, 
                      frame_num, gate_num, data_num, bed_num);
        proj_data->set_sinogram(sino_2D);           
      }
      
    } // end of 2D case
    
    cout<<endl;
  } // end of write
}


// takes a pre-allocated buffer (which will be modified)
void read_sinogram(Sinogram<float>& sino_2D,
                   char *buffer, 
		   FILE*fptr, 
		   int mat_index, 
		   int frame, int gate, int data, int bed)
{
  ECAT6_Main_header mhead;
  ScanInfoRec scanParams;
  const long matnum=
    cti_numcod (frame,mat_index,gate,data,bed);
  if (cti_read_ECAT6_Main_header (fptr, &mhead) != EXIT_SUCCESS) 
    error("read_sinogram: error reading ECAT6_Main_header");
  
  float scale_factor = 0; // intialised to avoid compiler warnings
  switch(mhead.file_type)
  {
    case matScanFile:
      {
        Scan_subheader shead;
        
        if(get_scanheaders (fptr, matnum, &mhead, 
                            &shead, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);
        
        scale_factor = shead.scale_factor;
        if (shead.loss_correction_fctr>0)
          scale_factor *= shead.loss_correction_fctr;
        else
          warning("\nread_sinogram warning: loss_correction_fctr invalid, using 1\n");
        break;
      }
    case matAttenFile:
      {
        Attn_subheader shead;
        
        if(get_attnheaders (fptr, matnum, &mhead, 
                            &shead, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);
        
        scale_factor = shead.scale_factor;
        break;
      }
    case matNormFile:
      {
        Norm_subheader shead;
        
        if(get_normheaders (fptr, matnum, &mhead, 
                            &shead, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);
        
        scale_factor = shead.scale_factor;
        break;
      }
    default:
      error("read_sinogram: unsupported format");
  }
  if(get_scandata (fptr, buffer, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);

  cti_data_to_float_Array(sino_2D, buffer, scale_factor, scanParams.data_type);

}


Succeeded 
DiscretisedDensity_to_ECAT6(FILE *fptr,
                            DiscretisedDensity<3,float> const & density, 
			    const ECAT6_Main_header& mhead,
                            const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  

  DiscretisedDensityOnCartesianGrid<3,float> const & image =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(density);

   
  if (mhead.file_type!= matImageFile)
  {
    warning("DiscretisedDensity_to_ECAT6: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.file_type should be ImageFile\n",
            frame_num, gate_num, data_num, bed_num);
    return Succeeded::no;
  }
  if (mhead.num_planes!=image.get_length())
  {
    warning("DiscretisedDensity_to_ECAT6: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.num_planes should be %d\n",
            frame_num, gate_num, data_num, bed_num,image.get_length());
    return Succeeded::no;
  }
  const float voxel_size_z = image.get_grid_spacing()[1]/10;// convert to cm
  //const float voxel_size_y = image.get_grid_spacing()[2]/10;
  const float voxel_size_x = image.get_grid_spacing()[3]/10;
  if (fabs(mhead.plane_separation - voxel_size_z) > 1.E-4) 
  {
    warning("DiscretisedDensity_to_ECAT6: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.plane_separation should be %g\n",
            frame_num, gate_num, data_num, bed_num,voxel_size_z);
    return Succeeded::no;
  }
  
  
  Image_subheader ihead= img_zero_fill();
  
  const int min_z= image.get_min_index();
  const int min_y= image[min_z].get_min_index();
  const int min_x= image[min_z][min_y].get_min_index();
  
  const int z_size= image.get_length();
  const int y_size= image[min_z].get_length();
  const int x_size= image[min_z][min_y].get_length();
  
  const int plane_size= y_size * x_size;
  
  // Setup subheader params
#ifndef STIR_ORIGINAL_ECAT6
  ihead.data_type=ECAT_I2_little_endian_data_type;
  ihead.x_dimension= x_size;
  ihead.y_dimension= y_size;
  ihead.z_dimension= z_size;
  ihead.x_pixel_size= voxel_size_x;
  ihead.y_pixel_size= voxel_size_x;
  ihead.z_pixel_size= voxel_size_z;
  
  ihead.num_dimensions= 3;
  // STIR origin depends on the index range, but the index range is lost
  // after writing to file.
  // ECAT6 origin is somewhere in the middle of the image
  // WARNING this has to be consistent with reading
  if (image[0][0].get_min_index() != -(x_size/2) ||
      image[0][0].get_max_index() != -(x_size/2) + x_size - 1 ||
      image[0].get_min_index() != -(y_size/2) ||
      image[0].get_max_index() != -(y_size/2) + y_size - 1 ||
      image.get_min_index() != 0)
    {
      warning("DiscretisedDensity_to_ECAT6 is currently limited to input images in the standard STIR index range.\n"
	      "Data not written.");
      return Succeeded::no;
    }
  ihead.x_offset= image.get_origin().x()/10;
  ihead.y_offset= image.get_origin().y()/10;
  ihead.z_offset= image.get_origin().z()/10;
  shared_ptr<Scanner> scanner_ptr = find_scanner_from_ECAT6_Main_header(mhead);

  const float depth_of_interaction_factor =
    1 + 
    scanner_ptr->get_average_depth_of_interaction() /
    scanner_ptr->get_inner_ring_radius();
  // note: CTI uses shead.x_resolution instead of mhead.bin_size
  // but we don't have access to the sinogram here, and these 2 fields 
  // should be equal anyway.
  ihead.recon_zoom= 
    mhead.bin_size/voxel_size_x *
    scanner_ptr->get_default_num_arccorrected_bins()/
    float(image[0].size()) *
    depth_of_interaction_factor;

  ihead.decay_corr_fctr= 1;
#else // STIR_ORIGINAL_ECAT6
  ihead.data_type= mhead.data_type;
  ihead.dimension_1= x_size;
  ihead.dimension_2= y_size;
  ihead.slice_width= mhead.plane_separation;
  ihead.pixel_size= voxel_size_x;
  
  ihead.num_dimensions= 2;
  ihead.x_origin= image.get_origin().x()/10;
  ihead.y_origin= image.get_origin().y()/10;
  ihead.recon_scale= 1;
  ihead.decay_corr_fctr= 1;
  ihead.loss_corr_fctr= 1;
  ihead.ecat_calibration_fctr= 1;
  ihead.well_counter_cal_fctr=1;
#endif // STIR_ORIGINAL_ECAT6
  
  // make sure we have a large enough multiple of MatBLKSIZE
  int cti_data_size = plane_size*2;
  if (cti_data_size%MatBLKSIZE != 0)
    cti_data_size = ((cti_data_size/MatBLKSIZE)+1)*MatBLKSIZE;
  short *cti_data= new short[cti_data_size/2];
  Array<2,short> plane(image[min_z].get_index_range());
  
  for(int z=0; z<z_size; z++) { // loop on planes
    float scale_factor = 0;
    convert_array(plane, scale_factor, image[z+min_z]);
    ihead.image_min= plane.find_min();
    ihead.image_max= plane.find_max();
#ifndef STIR_ORIGINAL_ECAT6
    ihead.scale_factor= scale_factor==0 ? 1.F : scale_factor;
#else
    ihead.quant_scale= scale_factor==0 ? 1.F : scale_factor;
#endif // STIR_ORIGINAL_ECAT6
    
    for(int y=0; y<y_size; y++)
    {
      for(int x=0; x<x_size; x++)
        cti_data[y*x_size+x]= plane[y+min_y][x+min_x];
    } 
    
    // write data
    long matnum= cti_numcod(frame_num, z-min_z+1, gate_num, data_num, bed_num);
    if(cti_write_image(fptr, matnum, &mhead, &ihead, cti_data, cti_data_size)!=EXIT_SUCCESS) {
      warning("Unable to write image plane %d at (f%d, g%d, d%d, b%d) to file, exiting.\n",
               z-min_z+1, frame_num, gate_num, data_num, bed_num);
      delete[] cti_data;      
      return Succeeded::no;
    }
  } // end of loop on planes
  delete[] cti_data;  
  return Succeeded::yes;
}


Succeeded 
DiscretisedDensity_to_ECAT6(DiscretisedDensity<3,float> const & density, 
			    string const & cti_name, string const&orig_name,
			    const Scanner& scanner,
                            const int frame_num, const int gate_num, const int data_num, const int bed_num)
{  
  ECAT6_Main_header mhead;
  make_ECAT6_Main_header(mhead, scanner, orig_name, density);

  
  FILE *fptr= cti_create (cti_name.c_str(), &mhead);
  Succeeded result =
    DiscretisedDensity_to_ECAT6(fptr,
                            density, 
			    mhead,
                            frame_num, gate_num,data_num, bed_num);
  
  fclose(fptr);    
  return result;
}

Succeeded 
ProjData_to_ECAT6(FILE *fptr, ProjData const& proj_data, const ECAT6_Main_header& mhead,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num,
		  const bool write_2D_sinograms)
{
  if (mhead.file_type!= matScanFile)
  {
    warning("ProjData_to_ECAT6: converting (f%d, g%d, d%d, b%d)\n"
            "Main header.file_type should be ImageFile\n",
            frame_num, gate_num, data_num, bed_num);
    return Succeeded::no;
  }

  const int max_segment_num = 
    write_2D_sinograms
    ? 0
    :
      min(proj_data.get_max_segment_num(), -proj_data.get_min_segment_num())
;
  const int min_segment_num = - max_segment_num;

  {
    int num_planes = 0;
    for(int segment_num=min_segment_num;
	segment_num <= max_segment_num;
	++segment_num)
      num_planes+= proj_data.get_num_axial_poss(segment_num);
    
    if (mhead.num_planes!=num_planes)
      {
	warning("ProjData_to_ECAT6: converting (f%d, g%d, d%d, b%d)\n"
		"Main header.num_planes should be %d, but is %d\n",
		frame_num, gate_num, data_num, bed_num,num_planes, mhead.num_planes);
	if (mhead.num_planes<num_planes)
	  return Succeeded::no;
      }
  }
  
  Scan_subheader shead= scan_zero_fill();
  
  const int min_view= proj_data.get_min_view_num();
  const int min_bin= proj_data.get_min_tangential_pos_num();
  
  const int num_view= proj_data.get_num_views();
  const int num_bin= proj_data.get_num_tangential_poss();
  
  const int plane_size= num_view * num_bin;
  
  // Setup subheader params
#ifndef STIR_ORIGINAL_ECAT6
  shead.data_type=ECAT_I2_little_endian_data_type;
  shead.num_r_elements= num_bin;
  shead.num_angles= num_view;
#else
  shead.data_type= mhead.data_type;
  shead.dimension_1= num_bin;
  shead.dimension_2= num_view;
#endif // STIR_ORIGINAL_ECAT6
  shead.loss_correction_fctr= 1; 
  // find sample_distance
  {
    ProjDataInfoCylindricalArcCorr const * const
      proj_data_info_cyl_ptr =
      dynamic_cast<ProjDataInfoCylindricalArcCorr const * const>
      (proj_data.get_proj_data_info_ptr());
    if (proj_data_info_cyl_ptr==NULL)
    {
      warning("This is not arc-corrected data. Filling in default_bin_size from scanner \n");
#ifndef STIR_ORIGINAL_ECAT6
      shead.x_resolution= 
#else
      shead.sample_distance= 
#endif
        proj_data.get_proj_data_info_ptr()->get_scanner_ptr()->get_default_bin_size()/10;
    }
    else
    {
#ifndef STIR_ORIGINAL_ECAT6
      shead.x_resolution= 
#else
      shead.sample_distance= 
#endif
        proj_data_info_cyl_ptr->get_tangential_sampling()/10;
    }
  }
    
  // find num_rings and check span
  int num_rings = proj_data.get_num_axial_poss(0);
  {
    ProjDataInfoCylindrical const * const
      proj_data_info_cyl_ptr =
      dynamic_cast<ProjDataInfoCylindrical const * const>
      (proj_data.get_proj_data_info_ptr());
    if (proj_data_info_cyl_ptr!=NULL)
      {
	// check if spanned data in segment 0
	if (proj_data_info_cyl_ptr->get_min_ring_difference(0) <
	    proj_data_info_cyl_ptr->get_max_ring_difference(0))
	  {
	    if (write_2D_sinograms)
	      num_rings = (proj_data.get_num_axial_poss(0)+1)/2;
	    else
	      {
		warning("Can only handle span==1 data. Exiting\n");
		return Succeeded::no;
	      }
	  }
      }
  }
	      

  if (num_rings != proj_data.get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings())
{
    warning("Expected %d num_rings from scanner while segment 0 implies %d rings\n",
            proj_data.get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings(), num_rings);
}
  
  short *cti_data= new short[plane_size];
  Array<2,short> short_sinogram(IndexRange2D(min_view,proj_data.get_max_view_num(),
    min_bin,proj_data.get_max_tangential_pos_num()));

  cout<<endl<<"Processing segment number:";
  
  for(int segment_num=min_segment_num;
      segment_num <= max_segment_num;
      ++segment_num)
  {    
    cout<<"  "<<segment_num;
    
    
    const int num_axial_poss= proj_data.get_num_axial_poss(segment_num);
    const int min_axial_poss= proj_data.get_min_axial_pos_num(segment_num);
    
    if (!write_2D_sinograms && num_axial_poss != num_rings - abs(segment_num))
    {
      warning("Can only handle span==1 data. Number of sinograms in this segment "
	      "should be %d. Exiting\n",  num_rings - abs(segment_num));
      delete[] cti_data;
      return Succeeded::no;
    }
    
    for(int z=0; z<num_axial_poss; z++) 
    { // loop on planes
      Sinogram<float> float_sinogram= proj_data.get_sinogram(z+min_axial_poss,segment_num,false);
      
      float scale_factor = 0;
      convert_array(short_sinogram, scale_factor, float_sinogram);
      
      
      shead.scan_min= short_sinogram.find_min();
      shead.scan_max= short_sinogram.find_max();
      shead.scale_factor= scale_factor==0 ? 1.F : scale_factor;
      
      for(int y=0; y<num_view; y++)
      {
        for(int x=0; x<num_bin; x++)
          cti_data[y*num_bin+x]= short_sinogram[y+min_view][x+min_bin];
      }         
      
      // write data
      int ring1, ring2;
      if (segment_num>=0)
      { ring1= z; ring2= z+segment_num; }
      else
      { ring1= z+abs(segment_num); ring2= z; }
      
      const int indexcod= 
	write_2D_sinograms 
	? z+1
	: cti_rings2plane( num_rings, ring1, ring2); // change indexation into CTI
      const long matnum= cti_numcod(frame_num, indexcod, gate_num, data_num, bed_num);
      if(cti_write_scan(fptr, matnum, &mhead, &shead, cti_data, plane_size*sizeof(short))!=EXIT_SUCCESS) 
      {
        warning("Unable to write short_sinogram for rings %d,%d to file, exiting.\n",ring1,ring2);
        delete[] cti_data;
        return Succeeded::no;
      }
    } // end of loop on planes
  } // end of loop on segments
  cout<<endl;
  delete[] cti_data;
  
  return Succeeded::yes;
}


Succeeded 
ProjData_to_ECAT6(ProjData const& proj_data, string const & cti_name, string const & orig_name,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num,
		  const bool write_2D_sinograms)
{  
  ECAT6_Main_header mhead;
  make_ECAT6_Main_header(mhead, orig_name, *proj_data.get_proj_data_info_ptr());
  
  
  FILE *fptr= cti_create(cti_name.c_str(), &mhead);   
  Succeeded result =
    ProjData_to_ECAT6(fptr, proj_data, mhead, 
		      frame_num, gate_num,data_num, bed_num, write_2D_sinograms);
  
  fclose(fptr);    
  return result;
}

void cti_data_to_float_Array(Array<2,float>&out, 
                             char const * const buffer, const float scale_factor, int dtype)
{

  BOOST_STATIC_ASSERT(sizeof(float)==4);

  switch (dtype)
  {
  case ECAT_Byte_data_type:
  {
    signed char const *  cti_data = 
      reinterpret_cast<signed char const * const >(buffer);
    for(int y=out.get_min_index(); y<=out.get_max_index(); y++)
      for(int x=out[y].get_min_index(); x<=out[y].get_max_index(); x++)
        out[y][x]=scale_factor*(*cti_data++);
    break;
  }
  case ECAT_I2_little_endian_data_type:
  case ECAT_I2_big_endian_data_type:
  {
    boost::int16_t const * cti_data = 
      reinterpret_cast<boost::int16_t const * const >(buffer);
    for(int y=out.get_min_index(); y<=out.get_max_index(); y++)
      for(int x=out[y].get_min_index(); x<=out[y].get_max_index(); x++)
        out[y][x]=scale_factor*(*cti_data++);
    break;
  }
  case ECAT_I4_little_endian_data_type:
  case ECAT_I4_big_endian_data_type:
    {
      boost::int32_t const * cti_data = 
	reinterpret_cast<boost::int32_t const * const >(buffer);
    for(int y=out.get_min_index(); y<=out.get_max_index(); y++)
      for(int x=out[y].get_min_index(); x<=out[y].get_max_index(); x++)
        out[y][x]=scale_factor*(*cti_data++);
    break;
  }
  case ECAT_R4_VAX_data_type:
  case ECAT_R4_IEEE_big_endian_data_type:
  {
    float const * cti_data = 
      reinterpret_cast<float const * const >(buffer);
    for(int y=out.get_min_index(); y<=out.get_max_index(); y++)
      for(int x=out[y].get_min_index(); x<=out[y].get_max_index(); x++)
        out[y][x]=scale_factor*(*cti_data++);
    break;
  }
  }
}

END_NAMESPACE_ECAT6
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
