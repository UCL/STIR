
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/phantoms/CylindersWithLineSource.h"
#include "local/stir/Shape/CombinedShape3D.h"
#include "stir/interfile.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/VectorWithOffset.h"
#include "stir/display.h"
#include "stir/shared_ptr.h"


#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
using std::ofstream;
using std::fstream;
#endif



USING_NAMESPACE_STIR
int
main() //int argc,char *argv[])
{
  
 /* shared_ptr<ProjDataFromStream> in_data=
    read_interfile_PDFS(argv[1]);

  const ProjDataInfo* data_info_ptr = in_data->get_proj_data_info_ptr();
  ProjDataInfo* data_info_ptr_cloned = in_data->get_proj_data_info_ptr()->clone();



  ProjDataInfoCylindrical* in_data_cyl = 
    dynamic_cast< ProjDataInfoCylindrical* >(data_info_ptr_cloned);

  ProjDataInfoCylindricalArcCorr* in_data_cyl_arcc = 
    dynamic_cast< ProjDataInfoCylindricalArcCorr* >(in_data_cyl);

  
  int min_view = in_data->get_min_view_num();
  int max_view = in_data->get_max_view_num();
  int min_tang = in_data->get_min_tangential_pos_num();
  int max_tang = in_data->get_max_tangential_pos_num();
  int min_segm_num  = in_data->get_min_segment_num();
  int max_segm_num  = in_data->get_max_segment_num();
  float bin_size = in_data_cyl_arcc->get_tangential_sampling();*/
/*
    Copyright (C) 2000- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/




  //ProjDataFromStream* in_data=
    //read_interfile_PDFS(argv[1]);
  
// const Scanner* scanner = in_data->get_proj_data_info_ptr()->get_scanner_ptr(); //(Scanner::RPT);
  
 CartesianCoordinate3D<float> origin(0,0,0);
 //CartesianCoordinate3D<float> voxel_size(2.425,2.25,2.25);
 CartesianCoordinate3D<float> voxel_size(1,1,1);


 
   /*VoxelsOnCartesianGrid<float> image1(IndexRange3D(0,67,-26,26,-26,26),origin,
     //-40,39,-40,40),origin,
     //-26,26),origin,
   voxel_size);
   
   VoxelsOnCartesianGrid<float> image2(IndexRange3D(0,67,-26,26,-26,26),origin,
   voxel_size);

   VoxelsOnCartesianGrid<float> image3(IndexRange3D(0,67,-26,26,-26,26),origin,
   voxel_size);*/

   VoxelsOnCartesianGrid<float> image1(IndexRange3D(0,1,-26,26,-26,26),origin,
   voxel_size);
   
   VoxelsOnCartesianGrid<float> image2(IndexRange3D(0,1,-26,26,-26,26),origin,
   voxel_size);

   VoxelsOnCartesianGrid<float> image3(IndexRange3D(0,1,-26,26,-26,26),origin,
   voxel_size);
 
 // for 966
   // VoxelsOnCartesianGrid<float> image2(IndexRange3D(0,94,-144,143,-144,143),origin,
  //  voxel_size);
 
  

   int min_x = image1.get_min_x();
   int max_x = image1.get_max_x();
   int min_y = image1.get_min_y();
   int max_y = image1.get_max_y();
   int min_z = image1.get_min_z();
   int max_z = image1.get_max_z();

  //CylindersWithLineSource_phantom CylindersWithLineSource;
   CylindersWithLineSource_phantom cylinderswithlinesource;
   CylindersWithLineSource_phantom cylinderswithlinesource_all;
   LineSource_phantom line_source;


  CartesianCoordinate3D<int> num_samples(3,2,2);
  CartesianCoordinate3D<float> offset(0,0,0);
  //CylindersWithLineSource.translate(offset);
  cylinderswithlinesource.translate(offset);
  line_source.translate(offset);

  
 // CylindersWithLineSource.get_A_ptr()->construct_volume(image, num_samples);
 // write_basic_interfile("test1", image);
  
 // CylindersWithLineSource.get_B_ptr()->construct_volume(image, num_samples);

  // here
  line_source.make_union_ptr(1)->construct_volume(image1, num_samples);
  //cylinderswithlinesource.make_union_ptr(1)->construct_volume(image2, num_samples);
  cylinderswithlinesource_all.make_union_ptr(1)->construct_volume(image3, num_samples);
  // for 966
  //cylinderswithlinesource.make_union_ptr(1)->construct_volume(image2, num_samples);


  
  for (int i = image1.get_min_z();i<=image1.get_max_z();i++)
  {
/*	image1[i][-1][-1] = 0.5;
	image1[i][-1][0]  = 0.5;
	image1[i][-1][1]  = 0.5;
	image1[i][0][-1]  = 0.5;
	image1[i][1][-1]  = 0.5;
	
	image1[i][0][0] = 0.5;
	image1[i][0][1] = 0.5;
	image1[i][1][0] = 0.5;
	image1[i][1][1] = 0.5;*/
	//image1[i][2][0] = 0.5;
	//image1[i][2][1] = 0.5;
	
	// off centre

	image3[i][0][4] = 0.5;
	image3[i][0][5]  = 0.5;
	image3[i][0][6]  = 0.5;
	image3[i][1][4]  = 0.5;
	image3[i][1][5]  = 0.5;
	
	
	image3[i][1][6] = 0.5;
	image3[i][2][4] = 0.5;
	image3[i][2][5] = 0.5;
	image3[i][2][6] = 0.5;
	
	/*image3[i][1][1] = 0.5;
	image3[i][1][2]  = 0.5;
	image3[i][1][3]  = 0.5;
	image3[i][2][1]  = 0.5;
	image3[i][2][2]  = 0.5;
	
	
	image3[i][2][3] = 0.5;
	image3[i][3][1] = 0.5;
	image3[i][3][2] = 0.5;
	image3[i][3][3] = 0.5;*/

	// for th eline source i the middle 
	/*image3[i][-1][-1] = 0.5;
	image3[i][-1][0]  = 0.5;
	image3[i][-1][1]  = 0.5;
	image3[i][0][-1]  = 0.5;
	image3[i][1][-1]  = 0.5;
	
	
	image3[i][0][0] = 0.5;
	image3[i][0][1] = 0.5;
	image3[i][1][0] = 0.5;
	image3[i][1][1] = 0.5;*/

	// for delta functions
//	image3[i][0][0]  = static_cast<float> (0.501);
//	image3[i][14][0] = static_cast<float> (1.1);
//	image3[i][7][0] = static_cast<float> (1.1);
//	image3[i][14][3] = static_cast<float> (1.001);
//	image3[i][10][3] = static_cast<float> (1.001);
	
  }

  /*for (int i = image1.get_min_z();i<=image1.get_max_z();i++)
  {
	image3[i][0][0]  = static_cast<float> (0.51);

  }

 for (int i = image3.get_min_z();i<=image3.get_max_z()-18;i++)
  {
  	image3[i][14][7] = static_cast<float> (1.01);
	image3[i][-13][-6] = static_cast<float> (1.01);
  }*/
 
  // for the line source off centre
  /*image3[0][2][2]  = static_cast<float> (0.51); 
  image3[1][2][2]  = static_cast<float> (0.51); 
  image3[0][14][7] = static_cast<float> (1.01);
  image3[0][-13][-6] = static_cast<float> (1.01);*/


  /*image3[0][1][5]   = static_cast<float> (0.51); 
  image3[1][1][5]   = static_cast<float> (0.51); 
  image3[0][14][7]  = static_cast<float> (1.01);
  image3[0][-13][1] = static_cast<float> (1.01);*/
  
  // new for dav and mrp
  /*image3[0][1][5]   = static_cast<float> (0.6); 
  image3[1][1][5]   = static_cast<float> (0.6); 
  image3[0][14][7]  = static_cast<float> (1.1);
  image3[0][-13][1] = static_cast<float> (1.1);*/




  /*image1[0][1][5]  = static_cast<float> (0.01); 
  image1[1][1][5]  = static_cast<float> (0.01); 
  image1[0][14][7] = static_cast<float> (0.01);
  image1[0][-13][1] = static_cast<float> (0.01);*/



  // to add the background  - cylinder -> add 0.0096
  
 /* for (int k = image2.get_min_z();k<=image3.get_max_z();k++) 
    for (int j = image2.get_min_y();j<=image3.get_max_y();j++) 
      for (int i = image2.get_min_x();i<=image3.get_max_x();i++) 
      {
	CartesianCoordinate3D<float> r (k,j,i);
	if(sqrt(inner_product(r,r)) <= 22)
	{
	  //if(image3[k][j][i]==0.F)
	  image2[k][j][i] = 0.96F;
	    //else 
	  //continue;
	}
      }*/


    /*  for (int k =0;k<=0;k++) 
	for (int j = image3.get_min_y();j<=image3.get_max_y();j++) 
	  for (int i = image3.get_min_x();i<=image3.get_max_x();i++) 
	  {
	    CartesianCoordinate3D<float> index (k,j,i);
	    CartesianCoordinate3D<float> origin1(0,14,4);
            CartesianCoordinate3D<float> dir_z (1,0,0);
	    CartesianCoordinate3D<float> dir_y (0,1,0);
	    CartesianCoordinate3D<float> dir_x (0,0,1);
	    
	    const CartesianCoordinate3D<float> r1 = index - origin1;
	    const float distance_along_axis1=
	      inner_product(r1,dir_z);
	     
	    
	    if (fabs(distance_along_axis1)<10/2)
	    { 
	      if ((square(inner_product(r1,dir_x))/square(8) + 
		square(inner_product(r1,dir_y))/square(8))<=1)
		image3[k][j][i]=0.165F;
		else 
		continue;
	    }
	  
	  }
      for (int k =0;k<=0;k++) 
	for (int j = image3.get_min_y();j<=image3.get_max_y();j++) 
	  for (int i = image3.get_min_x();i<=image3.get_max_x();i++) 
	  {
	    CartesianCoordinate3D<float> index (k,j,i);
	    CartesianCoordinate3D<float> origin2(0,-14,4);
            CartesianCoordinate3D<float> dir_z (1,0,0);
	    CartesianCoordinate3D<float> dir_y (0,1,0);
	    CartesianCoordinate3D<float> dir_x (0,0,1);
	    
	    const CartesianCoordinate3D<float> r2 = index - origin2;
	    
	    const float distance_along_axis2=
	      inner_product(r2,dir_z);	    
	    
	    if (fabs(distance_along_axis2)<10/2)
	    { 
	      if ((square(inner_product(r2,dir_x))/square(8) + 
		square(inner_product(r2,dir_y))/square(8))<=1)
		image3[k][j][i]=0.165F;
		else 
		continue;
	    
	    } 
	  }*/
	  
     /* for (int k = image3.get_min_z();k<=image3.get_max_z();k++) 
	for (int j = image3.get_min_y();j<=image3.get_max_y();j++) 
	  for (int i = image3.get_min_x();i<=image3.get_max_x();i++) 
	  {
	    CartesianCoordinate3D<float> index (k,j,i);
	    CartesianCoordinate3D<float> origin (k,1,5);

	    const CartesianCoordinate3D<float> r = index - origin;

	    // warning it is assumed that dir-z is along scanner axis
	    CartesianCoordinate3D<float> dir_z (1,0,0);
	    CartesianCoordinate3D<float> dir_y (0,1,0);
	    CartesianCoordinate3D<float> dir_x (0,0,1);
	    
	    
	    const float distance_along_axis=
	      inner_product(r,dir_z);

	    float length =10.F;
	    // ellipsoid
	    if (fabs(distance_along_axis)<length/2)
	    { 
	      if ((square(inner_product(r,dir_x))/square(15) + 
		square(inner_product(r,dir_y))/square(22))<=1)
		image2[k][j][i] = 0.96F;

	   //	image2[k][j][i] = 1.3F;
	      else
	   	continue;
	    }
	  
	  
	  }*/

  /*image3[0][1][5]  = static_cast<float> (0.0001+0.096); 
  image3[1][1][5]  = static_cast<float> (0.0001+0.096); 
  image3[0][14][7] = static_cast<float> (0.0001+0.165);
  image3[0][-13][1] = static_cast<float> (0.0001+0.165);*/

	 
  //write_basic_interfile("deltas", image1);
 // write_basic_interfile("att_image", image2);
  write_basic_interfile("cyl_wodeltas", image3);
  
  return EXIT_SUCCESS;
}
