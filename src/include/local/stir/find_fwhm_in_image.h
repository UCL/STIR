//
// $Id$
//
/*!
  \file
  \ingroup resolution
  \brief A collection of functions to measure resolution

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$

 */
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include <iostream>
#include <iomanip>
#include <list>
#include <algorithm>  
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cout;
using std::cin;
using std::cerr;
using std::min;
using std::max;
using std::setw;
#endif
using namespace std;

START_NAMESPACE_STIR
                          
/*!
   \ingroup resolution
   \brief 
   takes as input the Begin, the End and the Max_Element Iterators of a sequence of 
   numbers (e.g. vector) and the level of the maximum you want to have (e.g. maximum/2 or maximum/10). 
   It gives as output resolution in pixel size.   
*/
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& current_max_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height )  ; 
 
/*!
   \ingroup resolution
   \brief 
   takes as input the Begin, the End and the Max_Element Iterators of a sequence 
   of numbers (e.g. vector) and the level of the maximum you want to have (e.g. maximum/2 or maximum/10). 
   It gives as output resolution in pixel size.   
*/
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)  ;   
                                              
                        
/*!
   \ingroup resolution
   \brief 
   finds the maximum of the Input_Array and returns its 
   location as a vector in BasicCoordinate Field (only 3D implementation).
*/ 
template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,int>                        
maximum_location(const Array<num_dimensions,elemT>&);
/*!
   \ingroup resolution
   \brief 
   finds the maximum of the Input_Array in the given slice at the given
   dimension (z=1, y=2, x=3), and returns its location as a vector in BasicCoordinate Field 
   (only 3D implementation).
*/            
template<int num_dimensions,class elemT>
BasicCoordinate<num_dimensions,int>
maximum_location_per_slice(const Array<num_dimensions,elemT>& input_array,
                           const int slice, const int dimension);

/*!
   \ingroup resolution
   \brief
  extracts a column from the given array, that includes the particular voxel and 
  returns a column in Array<1,elemT> type, at the wanted dimension (z=1, y=2, x=3).
  {Only 3D implementation) 
*/ 
template <int num_dimensions, class elemT>
Array<1,elemT>
extract_line(const Array<num_dimensions,elemT> &,   
             BasicCoordinate<num_dimensions,int>, 
             const int dimension);  

/*!
   \ingroup resolution
   \brief
  interpolates a column from the given array, that includes the particular voxel and returns 
  a column in Array<1,elemT> type, at the wanted dimension (z=1, y=2, x=3). 

  It finds the real maximum location, using the 3 points parabolic fit. Then, linear interpolation is used  
  to find the whole line at the given dimension (z:1,y:2,x:3) taking into account the voxels that
  intersects the interpolated voxel which has in its center the point with the real_maximum_value.
  The do_direction is used so that in the direction of the line source will not happen any interpolation
  (Only 3D implementation).
*/ 
template <int num_dimensions, class elemT>
Array<1,elemT>
interpolated_line(const Array<num_dimensions,elemT>& input_array,    
                  const BasicCoordinate<num_dimensions,int>& max_location,
                  const BasicCoordinate<num_dimensions,bool>& do_direction, 
                  const int dimension);      
     
/*!
   \ingroup resolution
   \brief a structure that is used to hold the output of the function find_fwhm_in_image.
*/
template <int num_dimensions, class elemT>
struct ResolutionIndex
{
   elemT voxel_value;
   BasicCoordinate<num_dimensions, int> voxel_location;
   BasicCoordinate<num_dimensions, elemT> resolution;  
   BasicCoordinate<num_dimensions, elemT> real_maximum_value;      
};       
/*!
   \ingroup resolution
   \brief Finds FWHM for a number of point sources
 
   find_fwhm_in_image function 
   is the the basic function that the utility is based on. 
   It takes the 3D input_image, the number of maxima, the level of the maximum and 
   the dimension used for line sources.
   The implementation calculates and returns in a list container the maximum_value and its location per slice 
   for line sources or per the whole image for point sources. It calculates, as well, the resolution at the 
   same location, and the real_maximum_value linear 3D interpolation after 3 points parabolic fit.
   If the num_maximum>10, it uses the algorithm for line sources along the wanted dimension (z=1, y=2, x=3), 
   else for point sources. 
*/
template <int num_dimensions, class elemT>           
list<ResolutionIndex<num_dimensions,float> > 
find_fwhm_in_image(DiscretisedDensity<num_dimensions,elemT> & input_image,
                   const unsigned int num_maxima, const float level,const int dimension); 


END_NAMESPACE_STIR
