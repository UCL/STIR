//
// $Id$: $Date$
//
/*!
  \file

  \brief Implementation of routines which convert CTI things into our 
  building blocks and vice versa.

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/


#include "interfile.h"
#include "Sinogram.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
#include "IndexRange3D.h"
#include "CartesianCoordinate3D.h"
#include "VoxelsOnCartesianGrid.h"
#include "ByteOrder.h"
#include "ProjData.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "convert_array.h"
#include "IndexRange2D.h"
#include "utilities.h"

#include "Scanner.h" 
#include "CTI/cti_utils.h"
#include "CTI/Tomo_cti.h"

#include <iostream>
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ios;
#endif

START_NAMESPACE_TOMO

word find_CTI_system_type(const Scanner& scanner)
{
  switch(scanner.get_type())
  {
  case Scanner::E931:
    return 931; 
    
  case Scanner::E951:
    return 951; 
    
  case Scanner::E953:
    return 953;

  case Scanner::E961:
    return 961;

  case Scanner::E966:
    return 966;

  case Scanner::RPT:
    return 128;
    
  default:
    warning("find_CTI_system_type: %s currently unsupported\n. Defaulting to 953.\n", 
      scanner.get_name().c_str());
    return 953;
  }
}


Scanner* find_scanner_from_ECAT6_main_header(const Main_header& mhead)
{
  Scanner * scanner_ptr = 0;
  switch(mhead.system_type)
  {
  case 128 : 
    //camera = camRPT; 
    scanner_ptr = new Scanner(Scanner::RPT); 
    printf("Scanner : RPT\n"); 
    break;
  case 931 :
  case 12 : 
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
  default :  
    // enable our support for ECAT6 data for GE Advance
    if (mhead.num_planes == 324)
    {
      //camera = camAdvance; 
      scanner_ptr = new Scanner(Scanner::Advance); 
      printf("Scanner : GE Advance\n"); 
      break;
    }
    else
    {	
      //camera = camRPT; 
      scanner_ptr = new Scanner(Scanner::RPT); 
      printf("main_header.system_type unknown, defaulting to RPT \n"); 
    }
    break;
  }
  return scanner_ptr;
}

void make_ECAT6_main_header(Main_header& mhead, Scanner const * const scanner_ptr,
                            const string& orig_name                     
                            )
{
  mhead= main_zero_fill();
  mhead.calibration_factor = 1.F;
  
  // other header parameters
  
  strncpy(mhead.original_file_name, orig_name.c_str(), 20);
  mhead.num_frames= 1; //hdr.num_time_frames;
  // cti_utils routines always write data as VAX short
  mhead.data_type= matI2Data;
  
  mhead.system_type= find_CTI_system_type(*scanner_ptr);
  mhead.axial_fov= scanner_ptr->get_num_rings()*scanner_ptr->get_ring_spacing()/10;
  mhead.transaxial_fov= scanner_ptr->get_default_num_arccorrected_bins()*scanner_ptr->get_default_bin_size()/10;
  
  mhead.plane_separation= scanner_ptr->get_ring_spacing()/2/10;
  //WRONG mhead.gantry_tilt= scanner_ptr->get_default_intrinsic_tilt();
}


VoxelsOnCartesianGrid<float> *
ECAT6_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      FILE *cti_fptr, Main_header & mhead)
{
  MatDir entry;
  Image_subheader ihead;


  VoxelsOnCartesianGrid<float> * image_ptr = 0;
  
  // read first subheader to find dimensions
  {
    long matnum = cti_numcod(frame_num, 1,gate_num, data_num, bed_num);
    
    if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
      error("\nCouldn't find matnum %d in specified file.\n", matnum);            
    }
    
    if(cti_read_image_subheader(cti_fptr, entry.strtblk, &ihead)!=EXIT_SUCCESS)
      { 
	error("\nUnable to look up image subheader\n");
      }
  }
  
  const int x_size = ihead.dimension_1;
  const int y_size = ihead.dimension_2;
  const int z_size = mhead.num_planes;
  const int min_z = 0; 
  const int min_y = -y_size/2;
  const int min_x = -x_size/2;
  
  IndexRange3D range_3D (0,z_size-1,
			 -y_size/2,(-y_size/2)+y_size-1,
			 -x_size/2,(-x_size/2)+x_size-1);
  
  CartesianCoordinate3D<float> 
    voxel_size(ihead.slice_width*10,ihead.pixel_size*10,ihead.pixel_size*10);
  
  CartesianCoordinate3D<float> 
    origin(0, ihead.y_origin*10, ihead.x_origin*10);
  
  image_ptr = new VoxelsOnCartesianGrid<float> (range_3D, origin, voxel_size);
  
  // allocation for buffer
  short * cti_data= new short[x_size*y_size];
  
  // CTI stuff always assumes this
  assert(sizeof(short) == 2);

  for(int z=0; z<mhead.num_planes; z++) 
    { // loop upon planes
      long matnum = cti_numcod(frame_num, z+1,gate_num, data_num, bed_num);
        
      if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
	error("\nCouldn't find matnum %d in specified file.\n", matnum);     
      }
        
      if(cti_read_image_subheader(cti_fptr, entry.strtblk, &ihead)!=EXIT_SUCCESS) { // get ihead for plane z
	error("\nUnable to look up image subheader\n");
      }
        
      CartesianCoordinate3D<float> sub_head_origin(0, ihead.y_origin*10, ihead.x_origin*10);
      {
	if (image_ptr->get_origin() != sub_head_origin)
          {
            warning("ECAT6_to_VoxelsOnCartesianGrid: x,y offset of plane %d does not agree with plane 0. Ignoring it...\n",
		    z+1);
          }
      }

      float scale_factor = ihead.quant_scale;
      if (ihead.loss_corr_fctr>0)
	scale_factor *= ihead.loss_corr_fctr;
      else
	printf("\nread_plane warning: loss_corr_fctr invalid, using 1\n");

      if(cti_rblk (cti_fptr, entry.strtblk+1, cti_data, entry.endblk-entry.strtblk)!=EXIT_SUCCESS) { // get data
	error("\nUnable to read data\n");            
      }
 
      NumericType type;
      ByteOrder byte_order;
      find_type_from_cti_data_type(type, byte_order, ihead.data_type);
      if (!byte_order.is_native_order())
	swab((char *)cti_data, (char *)cti_data, (entry.endblk-entry.strtblk)*MatBLKSIZE); // swab the bytes
        
      for(int y=0; y<y_size; y++)
	for(int x=0; x<x_size; x++)
          {
            (*image_ptr)[z+min_z][y+min_y][x+min_x]=scale_factor*cti_data[y*x_size+x];
          }

    } // end loop on planes
    
  delete cti_data;
  return image_ptr;
}

void ECAT6_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      int max_ring_diff,
                      char *v_data_name, FILE *cti_fptr, Main_header &mhead)
{
  shared_ptr<Scanner> scanner_ptr = find_scanner_from_ECAT6_main_header(mhead);
  const int num_rings = scanner_ptr->get_num_rings();

  // ECAT 6 does not have a flag for 3D vs. 2D, so we guess it first from num_planes
  bool is_3D_file = (mhead.num_planes >  2*num_rings-1);
  if (!is_3D_file)
  {
    // better make sure by checking if plane (5,5) is not in its '3D' place
    ScanInfoRec scanParams;
    Scan_subheader shead;
    const int mat_index= cti_rings2plane(num_rings, 5,5); 
    const long matnum= cti_numcod(frame_num, mat_index,gate_num, data_num, bed_num);
    is_3D_file = (get_scanheaders(cti_fptr, matnum, &mhead, &shead, &scanParams)== EXIT_SUCCESS);
  }
  int span = 1;

  if (!is_3D_file)
  {
    warning("I'm guessing this is a 2D sinogram\n");
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
    warning("\n\aWarning: header says not enough planes in the file: %d (expected %d).\
    Continuing anyway...\n", mhead.num_planes, num_sinos);
  }
	
  // construct a ProjDataFromStream object
  shared_ptr<ProjDataFromStream>  proj_data =  NULL;

  {
    
    ScanInfoRec scanParams;
    Scan_subheader shead;
    // read first subheader for dimensions
    {     
      long matnum= cti_numcod(frame_num, 1,gate_num, data_num, bed_num);
      if (get_scanheaders(cti_fptr, matnum, &mhead, &shead, &scanParams)!= EXIT_SUCCESS)
        error("Error reading matnum %d\n", matnum);
    }   
    
    const int num_views = shead.dimension_2;
    const int num_tangential_poss = shead.dimension_1; 
    
    
    ProjDataInfo* p_data_info= 
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,span,max_ring_diff,num_views,num_tangential_poss); 
    
    
    ProjDataFromStream::StorageOrder  storage_order=
      ProjDataFromStream::Segment_AxialPos_View_TangPos;
    
    add_extension(v_data_name, ".s");
    iostream * sino_stream =
      new fstream (v_data_name, ios::out| ios::binary);
    
    if (!sino_stream->good())
    {
      error("ECAT6cti_to_PDFS: error opening file %s\n",v_data_name);
    }
    
    
    proj_data = 
      new ProjDataFromStream(p_data_info,sino_stream, streamoff(0), storage_order);
    
    write_basic_interfile_PDFS_header(v_data_name, *proj_data);
  }


  // write to proj_data
  {
    
    // allocation of buffer
    short* cti_data= 
      new short[proj_data->get_num_tangential_poss()*proj_data->get_num_views()];
    
    cerr<<"\nProcessing segment number:";
    
    if (is_3D_file)
    {
      for(int w=0; w<=max_ring_diff; w++) 
      { // loop on segment number
        
        // positive ring difference
        cerr<<"  "<<w;
        int num_axial_poss= num_rings-w;
        
        for(int ring1=0; ring1<num_axial_poss; ring1++) { // ring order: 0-0,1-1,..,15-15 then 0-1,1-2,..,14-15
          int ring2=ring1+w; // ring1<=ring2
          int mat_index= cti_rings2plane(num_rings, ring1, ring2);          
          Sinogram<float> sino_2D = proj_data->get_empty_sinogram(ring1,w,false);
          proj_data->set_sinogram(sino_2D);          
          read_sinogram(sino_2D, cti_data, cti_fptr, mat_index, 
            frame_num, gate_num, data_num, bed_num);
          proj_data->set_sinogram(sino_2D);                     
        }
        
        // negative ring difference
        if(w>0) {
          cerr<<"  "<<-w;
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
      cerr << "0\n";
      for(int z=0; z<proj_data->get_num_axial_poss(0); z++)
      {
        Sinogram<float> sino_2D = proj_data->get_empty_sinogram(z,0,false);
        read_sinogram(sino_2D, cti_data, cti_fptr, z+1, 
                      frame_num, gate_num, data_num, bed_num);
        proj_data->set_sinogram(sino_2D);           
      }
      
    } // end of 2D case
    
    cerr<<endl;
    delete cti_data;
  } // end of write
}


// takes a pre-allocated buffer (which will be modified)
void read_sinogram(Sinogram<float>& sino_2D,
                   short *buffer, 
		   FILE*fptr, 
		   int mat_index, 
		   int frame, int gate, int data, int bed)
{
  Main_header mhead;
  Scan_subheader shead;
  ScanInfoRec scanParams;
  const long matnum=
    cti_numcod (frame,mat_index,gate,data,bed);
  if(get_scanheaders (fptr, matnum, &mhead, 
                               &shead, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);
  
  float scale_factor = shead.scale_factor;
  if (shead.loss_correction_fctr>0)
    scale_factor *= shead.loss_correction_fctr;
  else
    printf("\nread_sinogram warning: loss_correction_fctr invalid, using 1\n");

  if(get_scandata (fptr, buffer, &scanParams)!= EXIT_SUCCESS)
          error("Error reading matnum %d\n", matnum);

  // copy from buffer to sino_2D
  const int min_tang_pos = sino_2D.get_min_tangential_pos_num();
  const int min_view=  sino_2D.get_min_view_num();
  
  for(int y=0; y<sino_2D.get_num_views(); y++)
    for(int x=0; x<sino_2D.get_num_tangential_poss(); x++)
    {
      sino_2D[y+min_view][x+min_tang_pos] =
        scale_factor* (*buffer++);
    }           
}


void find_type_from_cti_data_type(NumericType& type, ByteOrder& byte_order, const short data_type)
{
  switch(data_type)
  {
  case matByteData:
    type = NumericType("signed integer", 1);
    byte_order=ByteOrder::little_endian;
    return;
  case matI2Data:
    type = NumericType("signed integer", 2);
    byte_order=ByteOrder::little_endian;
    return;
  case matSunShort:
    type = NumericType("signed integer", 2);
    byte_order = ByteOrder::big_endian;
    return;
  case matVAXR4Data:
    type = NumericType("float", 4);
    byte_order=ByteOrder::little_endian;
    return;
  case matStdR4:
    type = NumericType("float", 4);
    byte_order=ByteOrder::big_endian;
    return;
  case matI4Data:
    type = NumericType("signed integer", 4);
    byte_order=ByteOrder::little_endian;
    return;
  case matSunLong:
    type = NumericType("signed integer", 4);
    byte_order=ByteOrder::big_endian;
    return;    
  default:
    error("find_type_from_cti_data_type: unsupported data_type: %d", data_type);
    // just to avoid compiler warnings
    return;
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
      return matByteData;
    case 2:
      return byte_order==ByteOrder::big_endian ? matSunShort : matI2Data;
    case 4:
      return byte_order==ByteOrder::big_endian ? matSunLong : matI4Data;
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
      return byte_order==ByteOrder::big_endian ? matStdR4 : matVAXR4Data;
    default:
      {
        // write error message below
      }
    }
  }
  string number_format;
  size_t size_in_bytes;
  type.get_Interfile_info(number_format, size_in_bytes);
  error("find_cti_data_type: CTI does not support data type '%s' of %d bytes.\n",
    number_format.c_str(), size_in_bytes);
  // just to satisfy compilers
  return short(0);
}

Succeeded 
DiscretisedDensity_to_ECAT6(DiscretisedDensity<3,float> const & density, string const & cti_name, string const&orig_name,
                            const int frame_num, const int gate_num, const int data_num, const int bed_num)
{
  

  VoxelsOnCartesianGrid<float> const & image =
    dynamic_cast<VoxelsOnCartesianGrid<float> const&>(density);
  
  Main_header mhead;
  
  {
    shared_ptr<Scanner> scanner_ptr = Scanner::ask_parameters();
    make_ECAT6_main_header(mhead, scanner_ptr.get(), orig_name);    
  }

  
  // extra main parameters that depend on data type
  mhead.file_type= matImageFile;
  mhead.num_planes=image.get_z_size();
  mhead.plane_separation=image.get_voxel_size().z()/10; // convert to cm
  
  FILE *fptr= cti_create (cti_name.c_str(), &mhead);
  
  
  Image_subheader ihead= img_zero_fill();
  
  const int min_z= image.get_min_z();
  const int min_y= image.get_min_y();
  const int min_x= image.get_min_x();
  
  const int z_size= image.get_z_size();
  const int y_size= image.get_y_size();
  const int x_size= image.get_x_size();
  
  const int plane_size= y_size * x_size;
  
  // Setup subheader params
  ihead.data_type= mhead.data_type;
  ihead.dimension_1= x_size;
  ihead.dimension_2= y_size;
  ihead.slice_width= mhead.plane_separation;
  ihead.pixel_size= image.get_voxel_size().x()/10; // convert to cm
  
  ihead.num_dimensions= 2;
  ihead.x_origin= image.get_origin().x()/10;
  ihead.y_origin= image.get_origin().y()/10;
  ihead.recon_scale= 1;
  ihead.decay_corr_fctr= 1;
  ihead.loss_corr_fctr= 1;
  ihead.ecat_calibration_fctr= 1;
  ihead.well_counter_cal_fctr=1;
  
  short *cti_data= new short[plane_size];
  Array<2,short> plane(image[min_z].get_index_range());
  
  for(int z=0; z<z_size; z++) { // loop on planes
    float scale_factor = 0;
    convert_array(plane, scale_factor, image[z+min_z]);
    ihead.image_min= plane.find_min();
    ihead.image_max= plane.find_max();
    ihead.quant_scale= scale_factor==0 ? 1.F : scale_factor;
    
    for(int y=0; y<y_size; y++)
    {
      for(int x=0; x<x_size; x++)
        cti_data[y*x_size+x]= plane[y+min_y][x+min_x];
    } 
    
    // write data
    long matnum= cti_numcod(frame_num, z-min_z+1, gate_num, data_num, bed_num);
    if(cti_write_image(fptr, matnum, &ihead, cti_data, plane_size*sizeof(short))!=EXIT_SUCCESS) {
      warning("Unable to write image plane %d to file, exiting.\n",z-min_z+1);
      delete[] cti_data;
      fclose(fptr);    
      return Succeeded::no;
    }
  } // end of loop on planes
  delete[] cti_data;
  fclose(fptr);    
  return Succeeded::yes;
}

Succeeded 
ProjData_to_ECAT6(ProjData const& proj_data, string const & cti_name, string const & orig_name,
                  const int frame_num, const int gate_num, const int data_num, const int bed_num)
{

    Scanner const * const scanner_ptr =
      proj_data.get_proj_data_info_ptr()->get_scanner_ptr();

    Main_header mhead;
    make_ECAT6_main_header(mhead, scanner_ptr, orig_name);

    // extra main parameters that depend on data type
    mhead.file_type= matScanFile;

    mhead.num_planes = 0;
    for(int segment_num=proj_data.get_min_segment_num();
        segment_num <= proj_data.get_max_segment_num();
        ++segment_num)
      mhead.num_planes+= proj_data.get_num_axial_poss(segment_num);

    FILE *fptr= cti_create(cti_name.c_str(), &mhead);   


    Scan_subheader shead= scan_zero_fill();

    const int min_view= proj_data.get_min_view_num();
    const int min_bin= proj_data.get_min_tangential_pos_num();
    
    const int num_view= proj_data.get_num_views();
    const int num_bin= proj_data.get_num_tangential_poss();
    
    const int plane_size= num_view * num_bin;
    
    // Setup subheader params
    shead.data_type= mhead.data_type;
    shead.dimension_1= num_bin;
    shead.dimension_2= num_view;
    shead.loss_correction_fctr= 1; 
    // find sample_distance
    {
      ProjDataInfoCylindricalArcCorr const * const
        proj_data_info_cyl_ptr =
        dynamic_cast<ProjDataInfoCylindricalArcCorr const * const>
        (proj_data.get_proj_data_info_ptr());
      if (proj_data_info_cyl_ptr==NULL)
      {
        warning("This does not seem to be arc-corrected data. Filling in default_bin_size from scanner \n");
        shead.sample_distance= scanner_ptr->get_default_bin_size();
      }
      else
      {
        shead.sample_distance= proj_data_info_cyl_ptr->get_tangential_sampling();
      }
    }
    
    short *cti_data= new short[plane_size];
    Array<2,short> short_sinogram(IndexRange2D(min_view,proj_data.get_max_view_num(),
      min_bin,proj_data.get_max_tangential_pos_num()));
    
    const int num_rings = proj_data.get_num_axial_poss(0);
    if (num_rings != scanner_ptr->get_num_rings())
      warning("Expected %d num_rings from scanner while segment 0 has %d planes\n",
      scanner_ptr->get_num_rings(), num_rings);
    
    
    cerr<<endl<<"Performing segment number:";
    
    for(int segment_num=proj_data.get_min_segment_num();
        segment_num <= proj_data.get_max_segment_num();
        ++segment_num)
    {
      
      
      cerr<<"  "<<segment_num;
      
      
      const int num_axial_poss= proj_data.get_num_axial_poss(segment_num);
      const int min_axial_poss= proj_data.get_min_axial_pos_num(segment_num);
      
      if (num_axial_poss != num_rings - abs(segment_num))
        error("Can only handle span==1 data. Exiting\n");
      
      for(int z=0; z<num_axial_poss; z++) 
      { // loop on planes
        Sinogram<float> float_sinogram= proj_data.get_sinogram(z+min_axial_poss,segment_num,false);
        
        float scale_factor = 0;
        convert_array(short_sinogram, scale_factor, float_sinogram);

#if 0
        {
          Sinogram<float> diff = float_sinogram;
          for(int y=0; y<num_view; y++)
          {
            for(int x=0; x<num_bin; x++)
              diff[y+min_view][x+min_bin]-= scale_factor*short_sinogram[y+min_view][x+min_bin];
          }         
          cerr << "plane "<< z << " scalef "<< scale_factor << " min " << diff.find_min() 
            << " max " << diff.find_max() << endl;
        }
#endif

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
        
        const int indexcod= cti_rings2plane( num_rings, ring1, ring2); // change indexation into CTI
        const long matnum= cti_numcod(frame_num, indexcod, gate_num, data_num, bed_num);
        if(cti_write_scan(fptr, matnum, &shead, cti_data, plane_size*sizeof(short))!=EXIT_SUCCESS) 
        {
          warning("Unable to write short_sinogram for rings %d,%d to file, exiting.\n",ring1,ring2);
          delete[] cti_data;
          fclose(fptr);    
          return Succeeded::no;
        }
      } // end of loop on planes
    } // end of loop on segments
    cerr<<endl;
    delete[] cti_data;
    fclose(fptr);  
    return Succeeded::yes;
}

END_NAMESPACE_TOMO
