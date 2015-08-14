//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2.0 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup utilities
 
  \brief  This program performs operations on image data

  \author Matthew Jacobson
  \author (with help from Kris Thielemans)
  \author PARAPET project


  \warning It only supports stir::VoxelsOnCartesianGrid type of images.
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/display.h"
#include "stir/utilities.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/recon_array_functions.h"
#include "stir/ArrayFunction.h"
#include "stir/zoom.h"


#include <iostream> 
#include <fstream>
#include <numeric>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ios;
using std::cerr;
using std::endl;
using std::swap;
#endif

USING_NAMESPACE_STIR

static void trim_edges(VoxelsOnCartesianGrid<float>& main_buffer);
static void get_plane(VoxelsOnCartesianGrid<float>& main_buffer);
static void get_plane_row(VoxelsOnCartesianGrid<float>& main_buffer);

static VoxelsOnCartesianGrid<float> ask_interfile_image(const char *const input_query);

static void show_menu();
static void show_math_menu();
static void math_mode(VoxelsOnCartesianGrid<float> &main_buffer, int &quit_from_math);

static VoxelsOnCartesianGrid<float> 
transpose_13(const VoxelsOnCartesianGrid<float> & image)
{
  CartesianCoordinate3D<float> origin = image.get_origin();
  swap(origin.x(), origin.z());
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  swap(voxel_size.x(), voxel_size.z());
  VoxelsOnCartesianGrid<float> 
    out(IndexRange3D(image.get_min_x(),image.get_max_x(),
                     image.get_min_y(),image.get_max_y(),
                     image.get_min_z(),image.get_max_z()),
        origin,
        voxel_size);
  for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
    for (int y=image.get_min_y(); y<=image.get_max_y(); ++y)
      for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
        out[x][y][z] = image[z][y][x];
  return out;
}

static VoxelsOnCartesianGrid<float> 
transpose_12(const VoxelsOnCartesianGrid<float> & image)
{
  CartesianCoordinate3D<float> origin = image.get_origin();
  swap(origin.y(), origin.z());
  CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
  swap(voxel_size.y(), voxel_size.z());
  VoxelsOnCartesianGrid<float> 
    out(IndexRange3D(image.get_min_y(),image.get_max_y(),
                     image.get_min_z(),image.get_max_z(),
                     image.get_min_x(),image.get_max_x()),
        origin,
        voxel_size);
  for (int y=image.get_min_y(); y<=image.get_max_y(); ++y)
    for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
      for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
        out[y][z][x] = image[z][y][x];
  return out;
}

int main(int argc, char *argv[])
{
    // file input
    VoxelsOnCartesianGrid<float> main_buffer;
    if(argc>1) 
    {
      main_buffer= 
	* dynamic_cast<VoxelsOnCartesianGrid<float> *>(
	DiscretisedDensity<3,float>::read_from_file(argv[1]));
    }
    else {
        cerr<<endl<<"Usage: manip_image <header file name> (*.hv)"<<endl<<endl;
        main_buffer= ask_interfile_image("File to load in main buffer? ");
    }

    int zs,ys,xs, ze,ye,xe,choice;

    zs=main_buffer.get_min_z();
    ys=main_buffer.get_min_y();
    xs=main_buffer.get_min_x(); 
  
    ze=main_buffer.get_max_z();  
    ye=main_buffer.get_max_y(); 
    xe=main_buffer.get_max_x();

    cerr<<"resx: "<<main_buffer.get_x_size()<<endl;
    cerr<<"resy: "<<main_buffer.get_y_size()<<endl;
    cerr<<"resz: "<<main_buffer.get_z_size()<<endl;
    cerr << "Min and Max in image " << main_buffer.find_min() 
         << " " << main_buffer.find_max() << endl;

    int plane=1,quit_from_math=0;

    show_menu();

    do { //start main mode
        choice = ask_num("Selection: ",0,13,13);

        switch(choice) {
            case 0: // quit
                break;

            case 1: // display
            {  
                const float maxi =
                  ask_num("Maximum in color scale",0.F,main_buffer.find_max(),main_buffer.find_max());
 
                switch (ask_num("transaxial (0), coronal (1), sagital (2)",0,2,0))
                {
                case 0: display(main_buffer, maxi); break;
                case 1: display(transpose_12(main_buffer), maxi); break;
                case 2: display(transpose_13(main_buffer), maxi); break;
                }
                break;
            }
            case 2: // data total
            {
                for (int z=zs; z<=ze; z++)
                    for (int y=ys; y <= ye; y++)
                        for (int x=xs; x<= xe; x++) cerr<<main_buffer[z][y][x]<<" ";
                break;
            }

            case 3: // data plane-wise
            {
                plane=1;
	
                while(plane>0 && plane <=ze-zs+1 ) {
                    plane = ask_num("Input plane # (0 to exit)", 0, ze-zs+1, zs);
	  
                    if(plane==0) break;	  	 
	  
                    for (int y=ys; y <= ye; y++)
                        for (int x=xs; x<= xe; x++)
                            cerr<<main_buffer[zs+plane-1][y][x]<<" "; // pause
                }
                break;
            }
 
            case 4: // min-max, counts plane-wise
            {
                plane=1;
	
                while(plane>0 && plane <=ze-zs+1 ) {
                    plane = ask_num("Input plane # (0 to exit)", 0, ze-zs+1, zs);
	  
                    if(plane==0) break;	  	 

                    cerr << "Min and Max in plane " 
                         << main_buffer[zs+plane-1].find_min() 
                         << " " << main_buffer[zs+plane-1].find_max() << endl;

                    cerr << "Number of counts = " 
                         << main_buffer[zs+plane-1].sum() 
                         << endl;
                }
                break;
            }

            case 5: // min-max image
            {
                cerr << "Min and Max in image " << main_buffer.find_min() 
                     << " " << main_buffer.find_max() << endl;
                break;
            }

            case 6: // trim
            {
                trim_edges(main_buffer);
                break;
            }

            case 7: // get plane
            {
                get_plane(main_buffer);
                break;
            }

            case 8: // get row
            {
                get_plane_row(main_buffer);
                break;
            }

            case 9: // counts
            {	
                cerr<<endl<<"The number of counts is: "<<main_buffer.sum()<<endl;      
                break;
            }
     
            case 10: //math mode
            {
                math_mode(main_buffer,quit_from_math);
                break;
            }

            case 11: //reload buffer from main menu
            {
                main_buffer = ask_interfile_image("File to load in buffer? ");
                break;
            }
    
            case 12: // buffer to file
            {
                char outfile[max_filename_length];
                ask_filename_with_extension(outfile, "Output filename (without extension) ", "");
                write_to_file(outfile, main_buffer);
                break;
            }  

            case 13: show_menu();
     
        } // end switch main mode
    } while(choice>0 && choice<=13 && (!quit_from_math));
    return EXIT_SUCCESS;
}

void trim_edges(VoxelsOnCartesianGrid<float>& input_image) 
{
    const int xe=input_image.get_max_x();
    const int xs=input_image.get_min_x();
    const int xm=(xs+xe)/2;
    const int rim_trunc=ask_num("By how many voxels to trim the FOV? ",0,(int)(xe-xm),2);

    truncate_rim(input_image,rim_trunc);

    if(ask("Zero end planes?",false)) truncate_end_planes(input_image);
}

void get_plane(VoxelsOnCartesianGrid<float>& input_image) 
{
    int zs,ys,xs, ze,ye,xe;

    zs=input_image.get_min_z();
    ys=input_image.get_min_y();
    xs=input_image.get_min_x(); 
  
    ze=input_image.get_max_z();  
    ye=input_image.get_max_y(); 
    xe=input_image.get_max_x();

    const int zm=(ze+zs)/2;

    int plane=ask_num("Which plane?",1,ze-zs+1,(int)zm-zs+1);

    ofstream profile;
    ask_filename_and_open(profile,  "Output filename ", ".prof", ios::out); 

    for (int y=ys; y<= ye; y++)
        for (int x=xs; x<= xe; x++)
            profile<<input_image[zs+plane-1][y][x]<<" ";
}

void get_plane_row(VoxelsOnCartesianGrid<float>& input_image) 
{
    int zs,ys,xs, ze,ye,xe;

    zs=input_image.get_min_z();
    ys=input_image.get_min_y();
    xs=input_image.get_min_x(); 
  
    ze=input_image.get_max_z();  
    ye=input_image.get_max_y(); 
    xe=input_image.get_max_x();

    float zm=(zs+ze)/2.F;
    float ym=(ys+ye)/2.F;
    float xm=(xs+xe)/2.F;
  
    int axdir=ask_num("Which axis direction (z=0,y=1,x=2)?",0,2,2);

    if (axdir==0) {
        int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);
        int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

        ofstream profile;
        ask_filename_and_open(profile,  "Output filename ", ".prof", ios::out); 

        for (int z=zs; z<= ze; z++)
            profile<<input_image[z][ycoord+ys-1][xcoord+xs-1]<<" ";
    }

    else if (axdir==1) {
        int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
        int xcoord=ask_num("X COORDINATE: ",1,xe-xs+1,(int)xm-xs+1);

        ofstream profile;
        ask_filename_and_open(profile,  "Output filename ", ".prof", ios::out); 

        for (int y=ys; y<= ye; y++)
            profile<<input_image[zcoord+zs-1][y][xcoord+xs-1]<<" ";
    }

    else { //axdir=2
        int zcoord=ask_num("Z COORDINATE: ",1,ze-zs+1,(int)zm-zs+1);
        int ycoord=ask_num("Y COORDINATE: ",1,ye-ys+1,(int)ym-ys+1);

        ofstream profile;
        ask_filename_and_open(profile,  "Output filename ", ".prof", ios::out); 

        for (int x=xs; x<= xe; x++)
            profile<<input_image[zcoord+zs-1][ycoord+ys-1][x]<<" ";
    }
}

VoxelsOnCartesianGrid<float> ask_interfile_image(const char *const input_query)
{
    char filename[max_filename_length];

    ask_filename_with_extension(filename, input_query, ".hv");

    shared_ptr<DiscretisedDensity<3,float> > 
      image_ptr(read_from_file<DiscretisedDensity<3,float> >(filename));
    return 
      * dynamic_cast<VoxelsOnCartesianGrid<float>*>(image_ptr.get());

}


void show_menu()
{
    cerr<<"\n\
MAIN MODE:\n\
0. Quit\n\
1. Display image\n\
2. Display all numeric data\n\
3. Display numeric data per plane\n\
4. Minimum, maximum & counts per plane \n\
5. Minimum & maximum in image\n\
6. Radial truncation \n\
7. Get plane\n\
8. Get row\n\
9. Counts\n\
10. Math mode\n\
11. Reload main buffer\n\
12. Write buffer to file\n\
13. Redisplay menu"<<endl;
}
void show_math_menu()
{
    cerr<<"\n\
MATH MODE:\n\
0. Quit \n\
1. Display math buffer\n\
2. Subtract image and take absolute value\n\
3. Add image\n\
4. Subtract image\n\
5. Multiply image\n\
6. Divide image (and truncate FOV)\n\
7. Add scalar\n\
8. Multiply scalar\n\
9. Divide scalar \n\
10. Zoom image \n\
11. Minimum & maximum in image\n\
12. Main buffer --> Math buffer\n\
13. Math buffer --> Main Buffer\n\
14. Reload main buffer\n\
15. Redisplay menu\n\
16. Main mode"<<endl;
}


void math_mode(VoxelsOnCartesianGrid<float> &main_buffer, int &quit_from_math)
{
    VoxelsOnCartesianGrid<float> math_buffer=main_buffer; //initialize math buffer
    int operation;

    show_math_menu();
  
    do {
        operation=ask_num("Choose Operation: ",0,16,15);

        switch(operation) { // math mode

            case 0: // quit
            { 
                quit_from_math=1;
                break;
            }

            case 1: // display math buffer
            {
                display(math_buffer, math_buffer.find_max());
                break;
            }

            case 2: // absolute difference
            {

                VoxelsOnCartesianGrid<float> aux_image=
                    ask_interfile_image("What image to compare with?");

                math_buffer-=aux_image;
                in_place_abs(math_buffer);

		//MJ 07/10/2000 removed trimming
                //trim_edges(math_buffer);

                cerr <<endl<< "Min and Max absolute difference " << math_buffer.find_min() 
                     << " " << math_buffer.find_max() << endl;
                cerr<<endl<<"Difference (L1 norm): "<<math_buffer.sum()<<endl;

                break;
            }

            case 3: // image addition
            {
                VoxelsOnCartesianGrid<float> aux_image=
                    ask_interfile_image("What image to add?");

                math_buffer+=aux_image;
                break;
            }


            case 4:  // image subtraction
            {
                VoxelsOnCartesianGrid<float> aux_image=
                    ask_interfile_image("What image to subtract?");

                math_buffer-=aux_image;
                break;
            }

            case 5:  // image multiplication
            {
                VoxelsOnCartesianGrid<float> aux_image=
                    ask_interfile_image("What image to multiply?");

                math_buffer*=aux_image;
                break;
            }

            case 6: // image division
            { 
                VoxelsOnCartesianGrid<float> aux_image= 
                    ask_interfile_image("What image to divide?");

                divide_array(math_buffer, aux_image);
                break;
            }

            case 7: //scalar addition
            {
                float scalar=ask_num("What scalar to add?", -100000.F,+100000.F,0.F);
    
                math_buffer+=scalar;
                break;
            }

            case 8: //scalar mulltiplication
            {
                float scalar=ask_num("What scalar to multiply?", -100000.F,+100000.F,1.F);
    
                math_buffer*=scalar;
                break;
            }

            case 9: //scalar division
            { 
                float scalar=0.0;

                do {
                    scalar=ask_num("What scalar to divide?", -100000.F,+100000.F,1.F);

                    if(scalar==0.0) cerr<<endl<<"Illegal -- division by 0"<<endl;

                } while(scalar==0.0);

                math_buffer/=scalar;
                break;
            }

            case 10: // zoom
	      {
                const float zoom_x = 
		  ask_num("Zoom factor x",0.1F,5.F,1.F);
                const float zoom_y = 
		  ask_num("Zoom factor y",0.1F,5.F,zoom_x);
                const float zoom_z = 
		  ask_num("Zoom factor z",0.1F,5.F,1.F);
                const float offset_x =
                  ask_num("Offset x (in mm)", 
			  -math_buffer.get_x_size()*math_buffer.get_voxel_size().x(),
			  math_buffer.get_x_size()*math_buffer.get_voxel_size().x(),
			  0.F);
                const float offset_y = 
		  ask_num("Offset y (in mm)", 
			  -math_buffer.get_y_size()*math_buffer.get_voxel_size().y(),
			  math_buffer.get_y_size()*math_buffer.get_voxel_size().y(),
			  0.F);
                const float offset_z = 
		  ask_num("Offset z (in mm)", 
			  -math_buffer.get_z_size()*math_buffer.get_voxel_size().z(),
			  math_buffer.get_z_size()*math_buffer.get_voxel_size().z(),
			  0.F);
                const int new_size_x = 
		  ask_num("New x size (pixels)", 1, 
			  static_cast<int>(math_buffer.get_x_size()*zoom_x * 2), 
			  static_cast<int>(math_buffer.get_x_size()*zoom_x));
                const int new_size_y = 
		  ask_num("New y size (pixels)", 1, 
			  static_cast<int>(math_buffer.get_y_size()*zoom_y * 2), 
			  new_size_x);
                const int new_size_z = 
		  ask_num("New z size (pixels)", 1, 
			  static_cast<int>(math_buffer.get_z_size()*zoom_z * 2), 
			  static_cast<int>(math_buffer.get_z_size()*zoom_z));
                zoom_image_in_place(math_buffer, 
				    CartesianCoordinate3D<float>(zoom_z, zoom_y, zoom_x),
				    CartesianCoordinate3D<float>(offset_z, offset_y, offset_x),
				    CartesianCoordinate3D<int>(new_size_z, new_size_y, new_size_x));
                break;
	      }

            case 11:
            {
                cerr << "Min and Max in image " << math_buffer.find_min() 
                     << " " << math_buffer.find_max() << endl;
                break;
            }

            case 12: //reinitialize math buffer
            {
                math_buffer=main_buffer;
                break;
            }

            case 13: //dump math buffer to main buffer
            {
                main_buffer=math_buffer;
                break;
            }

            case 14: //reload main buffer in math mode
            {
                main_buffer = ask_interfile_image("File to load in buffer? ");
                break;
            }

            case 15: //redisplay menu
            {
                show_math_menu();
                break;
            }


            case 16: //go back to main mode
            {
                show_menu();
            }

        } // end switch math mode
    } while(operation>0 && operation<16);
}
