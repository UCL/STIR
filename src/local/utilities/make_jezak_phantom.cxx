/*
    Copyright (C) 2000- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/Shape/EllipsoidalCylinder.h"
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

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
using std::ofstream;
using std::fstream;
#endif


START_NAMESPACE_STIR

VoxelsOnCartesianGrid<float>
make_cylinder (const float radius_a, const float radius_b,
   	       const CartesianCoordinate3D<float>& origin,
	       VoxelsOnCartesianGrid<float>& image);

VoxelsOnCartesianGrid<float>
make_cylinder (const float radius_a, const float radius_b,
   	       const CartesianCoordinate3D<float>& origin,
	       VoxelsOnCartesianGrid<float>& image)
{
  VoxelsOnCartesianGrid<float>* image_out = 
    image.get_empty_voxels_on_cartesian_grid();
   CartesianCoordinate3D<float> dir_z (1,0,0);
   CartesianCoordinate3D<float> dir_y (0,1,0);
   CartesianCoordinate3D<float> dir_x (0,0,1);

   for (int k = image.get_min_z();k<=image.get_max_z();k++) 
    for (int j = image.get_min_y();j<=image.get_max_y();j++) 
      for (int i = image.get_min_x();i<=image.get_max_x();i++) 
      {
	CartesianCoordinate3D<float> index (k,j,i);
	const CartesianCoordinate3D<float> r = index - origin;
	const float distance_along_axis=
	  inner_product(r,dir_z);
	
	if (fabs(distance_along_axis)<10)
	{ 
	  if ((square(inner_product(r,dir_x))/square(radius_a) + 
	    square(inner_product(r,dir_y))/square(radius_b))<=1)
	    (*image_out)[k][j][i] =1.F;
	  else continue;
	}
	else continue;
	
      }

      return *image_out;
}



/*
class Cylinder_phantom
{
public:

  inline Cylinder_phantom();
 
  inline Cylinder_phantom(const float radius_a, cont float radius_b,
			  const CartesianCoordinate3D<float>& origin);
 
  inline const shared_ptr<Shape3D>& get_A_ptr() const
  { return A_ptr; }

  inline shared_ptr<Shape3D> make_union_ptr(const float fraction) const;
   
  inline void translate(const CartesianCoordinate3D<float>& direction);
  inline void scale(const CartesianCoordinate3D<float>& scale3D);

private: 
  shared_ptr<Shape3D> A_ptr;
  
};


Cylinder_phantom::Cylinder_phantom()
{   
  
  A_ptr = new EllipsoidalCylinder (0,0,0,
    CartesianCoordinate3D<float>(0,0,0),
    0,0,0); 

}

Cylinder_phantom::Cylinder_phantom(const float radius_a, const float radius_b,
		      	           const CartesianCoordinate3D<float>& origin)
{

   A_ptr = new EllipsoidalCylinder (100,radius_a,radius_b,origin,
    0,0,1);

}

 
void 
Cylinder_phantom::translate(const CartesianCoordinate3D<float>& direction)
{
 A_ptr->translate(direction);   
}

void 
Cylinder_phantom::scale(const CartesianCoordinate3D<float>& scale3D)
{
  A_ptr->scale_around_origin(scale3D);
  }

shared_ptr<Shape3D>
Cylinder_phantom::make_union_ptr(const float fraction) const
{
   shared_ptr<Shape3D> full_A = get_A_ptr()->clone();
   
   return full_A;
    

}

*/
Array<1,float>
linespace (float start, float end, int num_points);

Array<1,float> 
linespace (float start, float end, int num_points)
{

  Array<1,float> linespaced;
  int length = end-start;
  if (num_points ==1)
  {
	  linespaced.grow(1,num_points);
	  linespaced[1] = start;
	  linespaced[num_points] = end;
  }
  else
  {
  if (length!=0)
  {
		float spacing = length/(num_points-1);
		linespaced.grow(1,num_points);
		linespaced[1] = start;
		int i =2;
		while (i <=num_points)
		{
		linespaced[i] = linespaced[i-1]+spacing;
		i++;    
		}
  }
  else
  {
  linespaced.grow(1,num_points);
  linespaced.fill(start);
  }
  }
  return linespaced;

}





END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int
main()
{

  /*
  Array <1,float> array1 (1,5);
  array1 = linespace(10,-10,5);

  for (int i = 1;i<=5;i++)
    cerr << array1[i] << "   ";*/
  


#if 1
  CartesianCoordinate3D<float> origin(0,48,48);
  CartesianCoordinate3D<float> voxel_size(1,1,1);
  
  VoxelsOnCartesianGrid<float>image(IndexRange3D(0,1,1,96,1,96),origin,  
    voxel_size);
  VoxelsOnCartesianGrid<float>image_out(IndexRange3D(0,1,1,96,1,96),origin,
    voxel_size);
  EllipsoidalCylinder cylinder;
  
  
  
  //Main disc radius (1 = to edge)
  float D = 3*9.7F;   
  
  //D = .92;
  //Z = disc(0,0,D,n);  
  
  // Rods diameters
  Array<1, float>  sz(1,6);
  //C-to-C spacing of rods
  Array <1,float> ts (1,6);

  ts[1] =sz[1] = 3*.8F;
  ts[2] =sz[2] = 3*1.1F;
  ts[3] =sz[3] = 3*1.3F;
  ts[4] =sz[4] = 3*1.5F;
  ts[5] =sz[5] = 3*1.8F;
  ts[6] =sz[6] = 3*2.5F;

  // very good
  /*ts[1] =sz[1] = 3*.8F;
  ts[2] =sz[2] = 3*1.1F;
  ts[3] =sz[3] = 3*1.2F;
  ts[4] =sz[4] = 3*1.4F;
  ts[5] =sz[5] = 3*1.8F;
  ts[6] =sz[6] = 3*2.4F;*/
  /*ts[1] =sz[1] = 3*.5F;
  ts[2] =sz[2] = 3*.7F;
  ts[3] =sz[3] = 3*.8F;
  ts[4] =sz[4] = 3*1.0F;
  ts[5] =sz[5] = 3*1.4F;
  ts[6] =sz[6] = 3*2.0F; */
  
  //additional extra spacing 
  Array <1,float> ex (1,6);
  ex.fill(2.5*.5);
  
  for (int j=1;j<=sz.get_length();j++)
  {
    float s  = sz[j];
    float t  = ts[j];
    float x1 = D*sin((double)j*60/180*_PI);    
    float y1 = D*cos((double)j*60/180*_PI);
    float x2 = D*sin((double)(j+1)*60/180*_PI);
    float y2 = D*cos((double)(j+1)*60/180*_PI);
    float x3 = (x1+x2)/3; 
    float y3 = (y1+y2)/3;
    
    //length of line seg connect arc
    float l = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));        
    // "extra" space at each end
    float e = (sqrt(3)-1)*s+ex[j];    
    //  Number of rods that fit
    float N = floor((l-2*e)/(s+t)+1);     
    //Actual space taken
    float S = (N-1)*(s+t);                        
    float b = S/l;
    
    float x4 = (x1-x3)*b+x3; 
    float y4 = (y1-y3)*b+y3;
    float x5 = (x2-x3)*b+x3; 
    float y5 = (y2-y3)*b+y3;
    float x6 = (-x3)*b+x3;  
    float y6 = (-y3)*b+y3;
    Array <1,float> rx1 (1,N);
    Array <1,float> ry1 (1,N);
    Array <1,float> rx2 (1,N);
    Array <1,float> ry2 (1,N);
    Array <1,float> rx  (1,N);
    Array <1,float> ry  (1,N);
    
    
    rx1 = linespace(x4,x5,N); 
    ry1 = linespace(y4,y5,N);
    rx2 = linespace(x4,x6,N);
    ry2 = linespace(y4,y6,N);
    for ( int k=1;k<=N;k++)
    {
      rx = linespace(rx1[k],rx2[k],k);
      ry = linespace(ry1[k],ry2[k],k);
      for (int jj=1; jj<=k ;jj++)
      { 
	CartesianCoordinate3D<float> dir_z (1,0,0);
	CartesianCoordinate3D<float> dir_y (0,1,0);
	CartesianCoordinate3D<float> dir_x (0,0,1);
	
	for (int  k= image.get_min_z();k<=image.get_max_z();k++) 
	  for (int j = image.get_min_y();j<=image.get_max_y();j++) 
	    for (int i = image.get_min_x();i<=image.get_max_x();i++) 
	    {
	      CartesianCoordinate3D<float> index (k,j,i);
	      const CartesianCoordinate3D<float> originl (k,ry[jj],rx[jj]);						
	      const CartesianCoordinate3D<float> r = origin -(index - originl);
	      const float distance_along_axis=
		inner_product(r,dir_z);

	      if(sqrt(inner_product(r,r)) <= s/2)
		{
		    image_out[k][j][i] = 0.10F;
	    	}
	      
	 
	    }

	for (int  k= image.get_min_z();k<=image.get_max_z();k++) 
	  for (int j = image.get_min_y();j<=image.get_max_y();j++) 
	    for (int i = image.get_min_x();i<=image.get_max_x();i++) 
	    {
	      CartesianCoordinate3D<float> index (k,j,i);
	      //const CartesianCoordinate3D<float> originl (0,ry[jj],rx[jj]);						
	      const CartesianCoordinate3D<float> r = index - origin;
	        if(sqrt(inner_product(r,r)) <= 30)
		{
		  if (image_out[k][j][i]==0.0F) 
		    image_out[k][j][i] = 1.0F;
	    	}
	    
	    
	    }
	    
	    //  }
	    //image_out= make_cylinder (s/2, s/2,
	    //CartesianCoordinate3D<float>(0,ry[j],rx[j]),image);
	    //}
      } 
      
    }
  }
  write_basic_interfile("jezak_phantom", image_out);

#endif
  return EXIT_SUCCESS;
  
  
  
				  
}
