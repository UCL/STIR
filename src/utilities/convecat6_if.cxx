//
// $Id$: $Date$
//

/*! 
  \file
  \ingroup utilities
  \brief Conversion from ecat 6 cti to interfile (image and sinogram data)
  \author Kris Thielemans
  \author Damien Sauge
  \author Sanida Mustafovic
  \author PARAPET project
  \version $Date$
  \date    $Revision$

  \warning Most of the data in the ECAT 6 headers is ignored (except dimensions)
  \warning Data are scaled using the subheader.scale_factor * subheader.loss_correction_fctr
  (unless the loss correctino factor is < 0, in which it is assumed to be 1).
  \warning Currently, the decay correction factor is ignored.
*/


#include "interfile.h"
#include "Sinogram.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfo.h"
#include "IndexRange3D.h"
#include "CartesianCoordinate3D.h"
#include "VoxelsOnCartesianGrid.h"
#include "ByteOrder.h"
#include "utilities.h"

#include "Scanner.h" 
#include "CTI/camera.h"
#include "CTI/cti_types.h"
#include "CTI/cti_utils.h"

#include <iostream>
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ios;
#endif



START_NAMESPACE_TOMO

/*
  \brief Convert image data
*/
// TODO mhead should be a const ref, but get_scanheaders needs to be modified first
static void ecat6cti_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      char *v_data_name, FILE *cti_fptr, Main_header & v_mhead, 
                      shared_ptr<Scanner> const& scanner_ptr,CameraType const& camera);
/* 
  \brief Convert sinogram data
  \param max_ring_diff if less than 0, the maximum is used (i.e. num_rings-1)
*/
// TODO mhead should be a const ref, but get_scanheaders needs to be modified first
static void ecat6cti_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      int max_ring_diff,
                      char *v_data_name, FILE *cti_fptr, Main_header & v_mhead, 
                      shared_ptr<Scanner> const& scanner_ptr,CameraType const& camera);

// determine scanner type from the main_header
static void find_scanner(shared_ptr<Scanner> & scanner_ptr,CameraType& camera, const Main_header& mhead);

/*
  \brief reads data from a CTI file into a Sinogram object
  \param buffer is supposed to be a pre-allocated buffer (which will be modified)

  It applies all scale factors.
  */
static void read_sinogram(Sinogram<float>& sino_2D,
                   short *buffer, 
		   FILE*fptr, CameraType camera, 
		   int mat_index, 
		   int frame, int gate, int data, int bed);

// Find out if we have to swap bytes after reading from the CTI file
static bool needs_swap_bytes(const short data_type);


void ecat6cti_to_VoxelsOnCartesianGrid(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      char *v_data_name, FILE *cti_fptr, Main_header & v_mhead, 
                      shared_ptr<Scanner> const& scanner_ptr,CameraType const& camera)
{
  MatDir entry;
  Image_subheader v_ihead;

  // read first subheader to find dimensions
  {
    long matnum = cti_numcod(camera, frame_num, 1,gate_num, data_num, bed_num);
    
    if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
      error("\nCouldn't find matnum %d in specified file.\n", matnum);            
    }
    
    if(cti_read_image_subheader(cti_fptr, entry.strtblk, &v_ihead)!=EXIT_SUCCESS) { 
      error("\nUnable to look up image subheader\n");
    }
  }
 
  const int x_size = v_ihead.dimension_1;
  const int y_size = v_ihead.dimension_2;
  const int z_size = v_mhead.num_planes;
  const int min_z = 0; 
  const int min_y = -y_size/2;
  const int min_x = -x_size/2;
  
  IndexRange3D range_3D (0,z_size-1,
	 -y_size/2,(-y_size/2)+y_size-1,
	 -x_size/2,(-x_size/2)+x_size-1);

    // KT 19/06/2000 reordered args correctly
  CartesianCoordinate3D<float> 
    voxel_size(v_ihead.slice_width,v_ihead.pixel_size,v_ihead.pixel_size);

  CartesianCoordinate3D<float> origin(0,0,0);

  VoxelsOnCartesianGrid<float> if_img3D(range_3D, origin, voxel_size);


  // allocation for buffer
  short * cti_data= new short[x_size*y_size];
  
  for(int z=0; z<v_mhead.num_planes; z++) { // loop upon planes
        long matnum = cti_numcod(camera, frame_num, z+1,gate_num, data_num, bed_num);
        
        if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
            error("\nCouldn't find matnum %d in specified file.\n", matnum);            
        }
        
        if(cti_read_image_subheader(cti_fptr, entry.strtblk, &v_ihead)!=EXIT_SUCCESS) { // get ihead for plane z
            error("\nUnable to look up image subheader\n");
        }
        
        float scale_factor = v_ihead.quant_scale;
        if (v_ihead.loss_corr_fctr>0)
          scale_factor *= v_ihead.loss_corr_fctr;
        else
          printf("\nread_plane warning: loss_corr_fctr invalid, using 1\n");

        if(cti_rblk (cti_fptr, entry.strtblk+1, cti_data, entry.endblk-entry.strtblk)!=EXIT_SUCCESS) { // get data
            error("\nUnable to read data\n");            
        }
   
	if (needs_swap_bytes(v_ihead.data_type))
          swab((char *)cti_data, (char *)cti_data, (entry.endblk-entry.strtblk)*MatBLKSIZE); // swab the bytes
        
	 for(int y=0; y<y_size; y++)
                for(int x=0; x<x_size; x++)
		{
                    if_img3D[z+min_z][y+min_y][x+min_x]=scale_factor*cti_data[y*x_size+x];
		}

    } // end loop on planes
    write_basic_interfile(v_data_name,if_img3D,voxel_size,NumericType::FLOAT);
    delete cti_data;
}

void ecat6cti_to_PDFS(const int frame_num, const int gate_num, const int data_num, const int bed_num,
                      int max_ring_diff,
                      char *v_data_name, FILE *cti_fptr, Main_header &v_mhead, 
                      shared_ptr<Scanner> const& scanner_ptr,const CameraType& camera)
{
  const int num_rings = scanner_ptr->get_num_rings();

  // ECAT 6 does not have a flag for 3D vs. 2D, so we guess it from num_planes
  const bool is_3D_file = (v_mhead.num_planes >  2*num_rings-1);
  int span = 1;

  if (!is_3D_file)
  {
    warning("I'm guessing this is a 2D sinogram\n");
    if(v_mhead.num_planes == 2*num_rings-1)
    {
      span=3; 
      max_ring_diff = 1;
    }
    else if (v_mhead.num_planes == num_rings)
    {
      span=1;
      max_ring_diff = 0;
    }
    else
    {
      error("Impossible num_planes: %d\n", v_mhead.num_planes);
    }
  }
  else
  {
    if (max_ring_diff < 0) 
      max_ring_diff = num_rings-1;
    const int num_sinos = 
      (2*max_ring_diff+1) * num_rings  - (max_ring_diff+1)*max_ring_diff;
    
    if (num_sinos > v_mhead.num_planes)
    warning("\n\aWarning: header says not enough planes in the file: %d (expected %d).\
    Continuing anyway...\n", v_mhead.num_planes, num_sinos);
  }
	
  // construct a ProjDataFromStream object
  shared_ptr<ProjDataFromStream>  proj_data =  NULL;

  {
    
    ScanInfoRec scanParams;
    Scan_subheader v_shead;
    // read first subheader for dimensions
    {     
      long matnum= cti_numcod(camera, frame_num, 1,gate_num, data_num, bed_num);
      if (get_scanheaders(cti_fptr, matnum, &v_mhead, &v_shead, &scanParams)!= EXIT_SUCCESS)
        error("Error reading matnum %d\n", matnum);
    }   
    
    const int num_views = v_shead.dimension_2;
    const int num_tangential_poss = v_shead.dimension_1; 
    
    
    ProjDataInfo* p_data_info= 
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,span,max_ring_diff,num_views,num_tangential_poss); 
    
    
    ProjDataFromStream::StorageOrder  storage_order=
      ProjDataFromStream::Segment_AxialPos_View_TangPos;
    
    add_extension(v_data_name, ".s");
    iostream * sino_stream =
      new fstream (v_data_name, ios::out| ios::binary);
    
    if (!sino_stream->good())
    {
      error("ecat6cti_to_PDFS: error opening file %s\n",v_data_name);
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
          read_sinogram(sino_2D, cti_data, cti_fptr, camera, mat_index, 
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
              cti_fptr, camera, mat_index, 
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
        read_sinogram(sino_2D, cti_data, cti_fptr, camera, z+1, 
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
		   FILE*fptr, CameraType camera, 
		   int mat_index, 
		   int frame, int gate, int data, int bed)
{
  Main_header mhead;
  Scan_subheader shead;
  ScanInfoRec scanParams;
  const long matnum=
    cti_numcod (camera,frame,mat_index,gate,data,bed);
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


static bool needs_swap_bytes(const short data_type)
{
  switch(data_type)
  {
  case matI2Data:
    return ByteOrder(ByteOrder::big_endian).is_native_order();
    
  case matSunShort:
    return ByteOrder(ByteOrder::little_endian).is_native_order();
    
  default:
    error("needs_swap_bytes: unsupported data_type: %d", data_type);
    // just to avoid compiler warnings
    return true;
  }
}

void find_scanner(shared_ptr<Scanner> & scanner_ptr,CameraType& camera, const Main_header& mhead)
{
  switch(mhead.system_type)
  {
  case 128 : 
    camera = camRPT; 
    scanner_ptr = new Scanner(Scanner::RPT); 
    printf("Scanner : RPT\n"); 
    break;
  case 931 : 
    camera = cam931; 
    scanner_ptr = new Scanner(Scanner::E931); 
    printf("Scanner : ECAT 931\n"); break;
  case 951 : 
    camera = cam951; 
    scanner_ptr = new Scanner(Scanner::E951); 
    printf("Scanner : ECAT 951\n"); break;
  case 953 : 
    camera = cam953; 
    scanner_ptr = new Scanner(Scanner::E953); 
    printf("Scanner : ECAT 953\n"); 
    break;
  default :  
    // enable our support for ECAT6 data for GE Advance
    if (mhead.num_planes == 324)
    {
      camera = camAdvance; 
      scanner_ptr = new Scanner(Scanner::Advance); 
      printf("Scanner : ECAT 951\n"); break;

      printf("Scanner : GE Advance\n"); 
    }
    else
    {	
      camera = camRPT; 
      scanner_ptr = new Scanner(Scanner::RPT); 
      printf("main_header.system_type unknown, defaulting to RPT \n"); 
    }
    break;
  }
}

END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO

int
main(int argc, char *argv[])
{
    char cti_name[max_filename_length], out_name[max_filename_length];
    FILE *cti_fptr;
 
    if(argc==2) 
      sprintf(cti_name,"%s",argv[1]);
    else {
        cerr<<"\nConversion from ECAT6 CTI data to interfile.\n";
        cerr<<"Usage: convecat6_if <data_file_name> (*.img / *.scn / *.*cti)\n"<<endl;
        ask_filename_with_extension(cti_name,"Name of the input data file? ",".scn");
        
    }
    ask_filename_with_extension(out_name,"Name of the output file? (.hv/.hs and .v/.s will be added)","");    

// open input file, read main header
    cti_fptr=fopen(cti_name, "rb"); 
    if(!cti_fptr) {
        error("\nError opening input file: %s\n",cti_name);
    }
    Main_header mhead;
    if(cti_read_main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
        error("\nUnable to read main header in file: %s\n",cti_name);
    }

    const int frame_num=ask_num("Frame number ? ",1,max(static_cast<int>(mhead.num_frames),1), 1);
    const int bed_num=ask_num("Bed number ? ",0,static_cast<int>(mhead.num_bed_pos), 0);
    const int gate_num=ask_num("Gate number ? ",1,max(static_cast<int>(mhead.num_gates),1), 1);
    const int data_num=ask_num("Data number ? ",0,8, 0);

    
    shared_ptr<Scanner> scanner_ptr;
    CameraType camera;
    
    find_scanner(scanner_ptr, camera, mhead);
    
    switch(mhead.file_type)
    { 
    case matImageFile:
      {                       
        ecat6cti_to_VoxelsOnCartesianGrid(frame_num, gate_num, data_num, bed_num,
                         out_name, cti_fptr, mhead, scanner_ptr,camera);
        break;
      }
    case matScanFile:
      {            
        const int max_ring_diff= 
           ask_num("Max ring diff to store (-1 == num_rings-1)",-1,100,-1);
        
        ecat6cti_to_PDFS(frame_num, gate_num, data_num, bed_num,
                         max_ring_diff, 
                         out_name, cti_fptr, mhead, scanner_ptr,camera);
        break;
      }
    default:
      {
        error("\nSupporting only image or scan file type at the moment. Sorry.\n");            
      }
    }    
    fclose(cti_fptr);
    
    return EXIT_SUCCESS;
}
