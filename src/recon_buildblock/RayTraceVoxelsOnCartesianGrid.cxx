//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementation of RayTraceVoxelsOnCartesianGrid  

  \author Kris Thielemans
  \author Mustapha Sadki
  \author (loosely based on some C code by Matthias Egger)
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "recon_buildblock/RayTraceVoxelsOnCartesianGrid.h"
#include "recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "CartesianCoordinate3D.h"
#include <math.h>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACE
using std::min;
using std::max;
#endif

START_NAMESPACE_TOMO

/*
  Siddon's algorithm works by looking at intersections of the 
  'intra-voxel' planes with the LOR.

  The LOR is parametrised as
  \code
  (x,y,z) = a (1/inc_x, 1/inc_y, 1/inc_z) + start_point
  \endcode
  Then values of 'a' are computed where the LOR intersects an intra-voxel plane.
  For example, 'ax' are the values where x= n + 0.5 (for integer n).
  Finally, we can go along the LOR and check which of the ax,ay,az is smallest,
  as this determines which plane the LOR intersects at this point.
*/
void 
RayTraceVoxelsOnCartesianGrid
        (ProjMatrixElemsForOneBin& lor, 
         const CartesianCoordinate3D<float>& start_point, 
         const CartesianCoordinate3D<float>& stop_point, 
         const CartesianCoordinate3D<float>& voxel_size,
         const float normalisation_constant)
{

  const CartesianCoordinate3D<float> difference = stop_point-start_point;

  assert(difference.x() <=0);
  assert(difference.y() >=0);
  assert(difference.z() >=0);

  // d12 is distance between the 2 points
  // it turns out we can multiply here with the normalisation_constant
  // (as that just scales the coordinate system)
  const float d12 = norm(difference*voxel_size) * normalisation_constant;
  
  
  const float inc_x = (difference.x()==0) ? 1000000.F : -d12 / difference.x();
  const float inc_y = (difference.y()==0) ? 1000000.F : d12 / difference.y();
  const float inc_z = (difference.z()==0) ? 1000000.F : d12 / difference.z();   

  // intersection points with  intra-voxel planes : 
  const float xmin = (int) (floor(start_point.x() + 0.5)) + 0.5;
  const float ymin = (int) (floor(start_point.y() - 0.5)) + 0.5;
  const float zmin = (int) (floor(start_point.z() - 0.5)) + 0.5;

  const float xmax = (int) (floor(stop_point.x() - 0.5)) + 0.5;
  const float ymax = (int) (floor(stop_point.y() + 0.5)) + 0.5;  
  const float zmax = (int) (floor(stop_point.z() + 0.5)) + 0.5;
  
  const float axend = (difference.x()==0) ? 1000000.F : (start_point.x() - xmax) * inc_x;
  const float ayend = (difference.y()==0) ? 1000000.F : (ymax - start_point.y()) * inc_y;
  const float azend = (difference.z()==0) ? 1000000.F : (zmax - start_point.z()) * inc_z;
  
  const float amax = min(axend, min(ayend, azend));
  
  // x,y,z-coordinates of the first Voxel: 
  int X = (int) (xmin - 0.5);
  int Y = (int) (ymin + 0.5);
  int Z = (int) (zmin + 0.5);
  
  // intersection point of the LOR :
  // with the previous xy-plane  (z smaller) : 
  float az = (difference.z()==0) ? -1. : (zmin - start_point.z()) * inc_z;
  // with the previous yz-plane (x smaller) : 
  float ax = (difference.x()==0) ? -1 : (start_point.x() - xmin) * inc_x;
  // with the previous xz-plane (y bigger) : 
  float ay = (difference.y()==0) ? -1 : (ymin - start_point.y()) * inc_y;
  
  // The biggest t?  value gives the start of the alpha-row 
  float a = max(ax, max(ay,az));      
  ax += inc_x;
  ay += inc_y;
  az += inc_z;
  
  {	  
    // go along the LOR 
    while ( a  < amax) {
      if ( ax < ay )    
        if (  ax  < az ) 
        { // LOR leaves voxel through yz-plane               	
          lor.push_back(ProjMatrixElemsForOneBin::value_type(Coordinate3D<int>(Z,Y,X),ax - a));
          a = ax;ax += inc_x;
          X--;
        }      	  
        else{ 	// LOR leaves voxel through xy-plane            	      
          lor.push_back(ProjMatrixElemsForOneBin::value_type(Coordinate3D<int>(Z,Y,X),az - a));	    
          a = az ;  az +=  inc_z;
          Z++; 
        } 
        else  if ( ay < az) {	// LOR leaves voxel through xz-plane 		                           
          lor.push_back(ProjMatrixElemsForOneBin::value_type(Coordinate3D<int>(Z,Y,X),ay - a));
          a = ay;   ay +=  inc_y;
          Y++;
        }  
        else {// LOR leaves voxel through xy-plane 			                      
          lor.push_back(ProjMatrixElemsForOneBin::value_type(Coordinate3D<int>(Z,Y,X),az - a ));
          a = az; az +=  inc_z;
          Z++; 
        }
    }	// end of while (a<amax)           
  }
}
END_NAMESPACE_TOMO
