//
// $Id$: $Date$
//

/*! 
  \file
  \ingroup utilities
  \brief Conversion from ecat 6 cti to interfile (image and sinogram data)
  \author Damien Sauge
  \author Sanida Mustafovic
  \author PARAPET project
  \version $Date$
  \date    $Revision$

  This program performs: <BR>
  - read and write the main header parameters <BR>
  - depending on the data type (scan-image): read and write the subheader parameters and the data
*/


#include "interfile.h"
#include "InterfileHeader.h"
#include "Sinogram.h"
#include "SegmentBySinogram.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfoCylindricalArcCorr.h"
#include "ProjDataInfo.h"
#include "IndexRange3D.h"
#include "BasicCoordinate.h"
#include "CartesianCoordinate3D.h"
//#include "StorageOrder.h"
#include "ByteOrder.h"
#include "NumericType.h"

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
using std::cin;
using std::ios;
#endif



START_NAMESPACE_TOMO
/*! 
  \brief Copy the CTI parameters into the interfile header
  \param scanner   Interfile scanner (scanner type, ring spacing, number of rings and views, ring radius, bin size, ...)
  \param camera    CTI scanner (camera type)
  \param v_mhead   CTI data header (file type, number of planes and frames, ...)
  \param ihead     CTI image subheader (dimensions, pixel size, reconstruction scale, ...)
  \param shead     CTI scan subheader (dimensions, pixel size, ...)
*/
void write_if_header(char *v_out_name, Main_header v_mhead, FILE *cti_fptr);
/*!
  \brief Convert image data
  \param data_file    output file
  \param v_data_name  name of the output file
  \param cti_data     1D array for CTI data (by planes)
  \param if_data      1D array for interfile data (by planes)

  For each planes: <BR>
  - read 1D CTI data <BR>
  - swab the bytes <BR>
  - write 1D interfile data into output file
*/
void ecat6cti_to_PIOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Image_subheader v_ihead,Scanner scanner,CameraType camera);
/*! 
  \brief Convert sinogram data
  \param data_file    output file
  \param v_data_name  name of the output file
  \param cti_data     1D array for CTI data (by ring difference)
  \param if_data      1D array for interfile data (by ring difference)

  For each segment: (segment order: 0,1,-1,2,-2,...,15,-15) <BR>
  - copy CTI data into interfile data for \b positive ring difference <BR>
  - copy CTI data into interfile data for \b negative ring difference
*/
void ecat6cti_to_PDFS(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Scan_subheader v_shead,Scanner scanner,CameraType camera);




void write_if_header(char *v_out_name, Main_header v_mhead, FILE *cti_fptr)
{
    char header_name[50]="",data_name[50]="", *syst_name;

    // use Unknown_Scanner to construct an object (TODO)
    Scanner scanner(Scanner::Unknown_Scanner);
    CameraType camera;
    MatDir entry;

    camera=camRPT; // defaulting to camRPT only to read sub headers

    long matnum=cti_numcod(camera,v_mhead.num_frames,v_mhead.num_planes,v_mhead.num_gates,0,v_mhead.num_bed_pos); // get matnum
    if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
        cerr<<endl<<"Couldn't find matnum "<<matnum<<" in specified file."<<endl;
        exit (EXIT_FAILURE);
    }
    
    switch(v_mhead.file_type)
    { 
// IMAGE DATA
        case matImageFile:
          {
            // KT 12/03/2000 moved locally
            Image_subheader ihead;

            if(cti_read_image_subheader(cti_fptr, entry.strtblk, &ihead)!=EXIT_SUCCESS) { // get ihead
                cerr<<endl<<"Unable to look up image subheader"<<endl;
                exit (EXIT_FAILURE);
            }

            // determine camera with dimension parameters (ihead), better than using mhead
            // TODO does not work !
            if(ihead.dimension_1<=129) {
                camera=camRPT;
                scanner=Scanner::RPT;
            }
            else if(ihead.dimension_1<=161) {
                camera=cam953;
                scanner=Scanner::E953;
            }
            else {
                camera=cam951;
                scanner=Scanner::E951;
            }

            sprintf(header_name, "%s.hv", v_out_name); 
            sprintf(data_name, "%s.v", v_out_name);

            // write interfile main header parameters  

            ecat6cti_to_PIOV(data_name, cti_fptr, v_mhead, ihead,scanner,camera);
            break;
          }
// SINOGRAM DATA
        case matScanFile:
          {
            // KT 12/03/2000 moved locally, substituted ihead with shead below
            Scan_subheader shead;

            if(cti_read_scan_subheader(cti_fptr, entry.strtblk, &shead)!=EXIT_SUCCESS) { // get shead
                cerr<<endl<<"Unable to look up image subheader"<<endl;
                exit (EXIT_FAILURE);
            }

            // determine camera with dimension parameters (shead), better than using mhead
            if(shead.dimension_1<=128) {
                camera=camRPT;
                scanner=Scanner::RPT;
                syst_name="PRT-1";
            }
            else if(shead.dimension_1<=160) {
                camera=cam953;
                scanner=Scanner::E953;
                syst_name="ECAT 953";
            }
            else {
                camera=cam951;
                scanner=Scanner::E951;
                syst_name="ECAT 951";
            }

            sprintf(header_name, "%s.hs", v_out_name);
            sprintf(data_name, "%s.s", v_out_name);

           // fclose(hdr_fptr);
            ecat6cti_to_PDFS(data_name, cti_fptr, v_mhead, shead,scanner,camera);
            //break;
          //}
        //default:
          //{
            //cerr<<endl<<"Unable to determine file type (either image or scan)."<<endl;
            //exit(EXIT_FAILURE);
            //break;
          }
    }
}

void ecat6cti_to_PIOV(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Image_subheader v_ihead,Scanner scanner,CameraType camera)
{
    MatDir entry;

    int dim_x= v_ihead.dimension_1;
    int dim_y= v_ihead.dimension_2;
    
    // allocation
    short *cti_data= (short *) calloc(dim_x*dim_y, sizeof(short));
    float *if_data= (float *) calloc(dim_x*dim_y, sizeof(float));
    	int x_size = v_ihead.dimension_1;
	int y_size = v_ihead.dimension_2;
	int z_size = v_mhead.num_planes;
	int min_z = 0; 
        int min_y = -y_size/2;
	int min_x = -x_size/2;

    	IndexRange3D range_3D (0,z_size-1,
	 -y_size/2,(-y_size/2)+y_size-1,
	 -x_size/2,(-x_size/2)+x_size-1);

     Array<3,float> if_img3D(range_3D);
     CartesianCoordinate3D<float> voxel_size(v_ihead.pixel_size,v_ihead.pixel_size,
	 v_ihead.slice_width);

    
    for(int z=0; z<v_mhead.num_planes; z++) { // loop upon planes
        long matnum = cti_numcod(camera, v_mhead.num_frames, z+1,v_mhead.num_gates,0,v_mhead.num_bed_pos);
        
        if(!cti_lookup(cti_fptr, matnum, &entry)) { // get entry
            cerr<<endl<<"Couldn't find matnum "<<matnum<<" in specified file."<<endl;
            exit (EXIT_FAILURE);
        }
        
        if(cti_read_image_subheader(cti_fptr, entry.strtblk, &v_ihead)!=EXIT_SUCCESS) { // get ihead for plane z
            cerr<<endl<<"Unable to look up image subheader"<<endl;
            exit (EXIT_FAILURE);
        }

        if(cti_rblk (cti_fptr, entry.strtblk+1, cti_data, entry.endblk-entry.strtblk)!=EXIT_SUCCESS) { // get data
            cerr<<endl<<"Unable to read data"<<endl;
            exit (EXIT_FAILURE);
        }
    
	swab((char *)cti_data, (char *)cti_data, (entry.endblk-entry.strtblk)*MatBLKSIZE); // swab the bytes
        
        for(int num_pixels=0; num_pixels<dim_y*dim_x; num_pixels++) 
	  if_data[num_pixels]= v_ihead.quant_scale*(float)cti_data[num_pixels];

	 for(int y=0; y<y_size; y++)
                for(int x=0; x<x_size; x++)
		{
                    if_img3D[z+min_z][y+min_y][x+min_x]=if_data[y*x_size+x];
		}

    } // end loop upon planes
     write_basic_interfile(v_data_name,if_img3D,voxel_size,
			   NumericType::FLOAT);
    free(cti_data);
}

void ecat6cti_to_PDFS(char *v_data_name, FILE *cti_fptr, Main_header v_mhead, 
                      Scan_subheader v_shead,Scanner scanner,CameraType camera)
{
  ScanInfoRec scanParams;
  
  int dim_bin= v_shead.dimension_1;
  int dim_view= v_shead.dimension_2;
  
  // allocation
  short *cti_data= (short *) calloc(dim_bin*dim_view, sizeof(short));
  float *if_data= (float *) calloc(dim_bin*dim_view, sizeof(float));
  
  const int num_views = dim_view;
  const int num_tangential_poss = dim_bin; 

  Scanner * scanner_ptr = new Scanner(scanner);

  
  const int num_rings = scanner_ptr->get_num_rings();
  
  // TODO allow for different max_delta

   ProjDataInfo* p_data_info= 
     ProjDataInfo::ProjDataInfoCTI(scanner_ptr,1,num_rings-1,num_views,num_tangential_poss); 

 
  ProjDataFromStream::StorageOrder  storage_order=ProjDataFromStream:: Segment_AxialPos_View_TangPos;        
  float scale_factor = 1 ;
  
  iostream * sino_stream;
  sino_stream = new fstream (v_data_name, ios::out| ios::binary);

  if (!sino_stream->good())
  {
    error(":ecat6cti_to_PDFS: error opening file %s\n",v_data_name);
  }
  
  
  ProjDataFromStream*  proj_data = new ProjDataFromStream(
    p_data_info,sino_stream,streamoff(0),
    storage_order,
    NumericType::FLOAT,
    ByteOrder::native,scale_factor);
  
  
  
  cerr<<endl<<"Processing segment number:";
  for(int w=0; w<num_rings; w++) { // loop on segment number
    
    // positive ring difference
    cerr<<"  "<<w;
    int num_ring= num_rings-w;
    // KT 13/03/2000 interchanged
    for(int ring1=0; ring1<num_ring; ring1++) { // ring order: 0-0,1-1,..,15-15 then 0-1,1-2,..,14-15
      int ring2=ring1+w; // ring1<=ring2
      int mat_index= cti_rings2plane(num_rings, ring1, ring2);
      long matnum= cti_numcod(camera, v_mhead.num_frames, mat_index,v_mhead.num_gates,0,v_mhead.num_bed_pos);
      get_scanheaders(cti_fptr, matnum, &v_mhead, &v_shead, &scanParams); // get scanParams
      get_scandata(cti_fptr, cti_data, &scanParams); // read data
      
      for(int num_pixels=0; num_pixels<dim_view*dim_bin; num_pixels++) 
	if_data[num_pixels]= v_shead.scale_factor*(float)cti_data[num_pixels];
          
      
      Sinogram<float> sino_2D = proj_data->get_proj_data_info_ptr()->get_empty_sinogram(ring1,w,false);
      
      int min_tang_pos = sino_2D.get_min_tangential_pos_num();
      int min_view=  sino_2D.get_min_view_num();
         
      for(int y=0; y<dim_view; y++)
	for(int x=0; x<dim_bin; x++)
	{
	  sino_2D[y+min_view][x+min_tang_pos] =if_data[y*dim_bin+x];

	} 

	proj_data->set_sinogram(sino_2D);
	
	
    }
    
    // negative ring difference
    if(w>0) {
      cerr<<"  "<<-w;
      for(int ring2=0; ring2<num_ring; ring2++) { // ring order: 0-1,2-1,..,15-14 then 2-0,3-1,..,15-13
	int ring1=ring2+w; // ring1>ring2
	int mat_index= cti_rings2plane(num_rings, ring1, ring2);
	long matnum= cti_numcod(camera, v_mhead.num_frames, mat_index,v_mhead.num_gates,0,v_mhead.num_bed_pos);
	get_scanheaders(cti_fptr, matnum, &v_mhead, &v_shead, &scanParams); // get scanParams
	get_scandata(cti_fptr, cti_data, &scanParams); // read data
	
	for(int num_pixels=0; num_pixels<dim_view*dim_bin; num_pixels++) 
	  if_data[num_pixels]= v_shead.scale_factor*(float)cti_data[num_pixels];
	
	Sinogram<float> sino_2D = proj_data->get_proj_data_info_ptr()->get_empty_sinogram(ring2,w,false);
	
	int min_tang_pos = sino_2D.get_min_tangential_pos_num();
	int min_view=  sino_2D.get_min_view_num();
	
	for(int y=0; y<dim_view; y++)
	  for(int x=0; x<dim_bin; x++)
	  {
	   sino_2D[y+min_view][x+min_tang_pos] =if_data[y*dim_bin+x];
	   
	  } 
	  
	 
	  proj_data->set_sinogram(sino_2D);
	  
      }
    }
  } // end of loop on segment number
  cerr<<endl;
  write_basic_interfile_PDFS_header(v_data_name,
			    *proj_data);
  delete sino_stream;
  free(cti_data);
}

END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO

main(int argc, char *argv[])
{
    char cti_name[50], out_name[50], data_name[50]="";
    FILE *cti_fptr;
 
    if(argc==2) sprintf(cti_name,"%s",argv[1]);
    else {
        cerr<<endl<<"Conversion from ECAT6 CTI data to interfile."<<endl;
        cerr<<"Usage: convecat6_if <data_file_name> (*.img / *.scn / *.*cti)"<<endl<<endl;
        system("ls *.scn *.img *.s2cti *.v2cti");
        cerr<<endl<<"Name of the input data file? : ";
        cin>>cti_name;
    }
    cerr<<endl<<"Name of the output file? (.hv/.hs and .v/.s will be added): ";
    cin>>out_name;

// open input file, read main header
    cti_fptr=fopen(cti_name, "rb"); 
    if(!cti_fptr) {
        cerr<<endl<<"Error opening input file: "<<cti_name<<endl;
        exit(EXIT_FAILURE);
    }
    Main_header mhead;
    if(cti_read_main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
        cerr<<endl<<"Unable to read main header in file: "<<cti_name<<endl;
        exit(EXIT_FAILURE);
    }

    write_if_header(out_name, mhead, cti_fptr);
    fclose(cti_fptr);
    // KT 12/03/2000 added main return value
    return EXIT_SUCCESS;
}
