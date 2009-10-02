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
#include <list>

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
                       const float level_height)  ;  
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
};       
/*!
   \ingroup resolution
   \brief Finds FWHM in pixel size units for a number of point sources or a line source   
  
   is the basic function that the utility is based on. It takes the 3D input_image, the number of maxima, 
   the level of the maximum (2 as half maximum, 10 as tenth maximum etc.), the dimension used for line 
   sources and a boolean nema which enables the calculation based on the NEMA Standards Publication 
   NU 2-2001. The implementation calculates and returns in a list container the maximum_value and its 
   location per slice for line sources, sorted by minimum to maximum slice of the using dimension, or per 
   whole image for point sources, sorted by maximum to minimum value. It calculates, as well, the resolution 
   at the same location in the size of units that is used (usually mm). If the dimension is set to 0 it finds 
   the resolution of each one point sources, else it finds the algorithm for the line sources giving its 
   direction (z=1, y=2, x=3), respectively. If given as [num_maxima] less than the total slices it returns 
   the results for some slices by sampling with the same step the total slices of the wanted dimension. 
   (Only 3D implementation). In case it is not needed to be followed to the NEMA standard, for better 
   approximation could be achieved by setting the nema to 0. 
   (In this case the interpolated_line function is used). 
*/

template <int num_dimensions, class elemT>           
std::list<ResolutionIndex<num_dimensions,float> > 
find_fwhm_in_image(DiscretisedDensity<num_dimensions,elemT> & input_image,
                   const unsigned int num_maxima, const float level, 
                   const int dimension, const bool nema);
/*!  
   \ingroup resolution
   \brief                              
   Masks the input_array

  it masks the input_array at the location of a point source setting all values to -1, with a mask size 
  depending on the resolution of the previous point (in pixel units), at each of the directions. In the 
  code there is a scale factor to change easily the size of the mask.
  (mask_size(z,y,x)=int(6*resolution(z,y,x)/level)). 
*/
template <int num_dimensions, class elemT>   
void 
flexible_mask(Array<num_dimensions,elemT>& input_array, 
              const BasicCoordinate<num_dimensions,int>& max_location,
              const BasicCoordinate<num_dimensions,elemT>& resolution,
              const float level);
/*!  
   \ingroup resolution
   \brief     
   Calculates the maximum point (x0,y0) of a parabola that passes through 3 points.

   As input it takes the Begin and End Iterators of a sequence of numbers (e.g. vector). 
   The three points are the maximum (x2,y2) of this sequence and the two neihbour points 
   (x1,y1) and (x3,y3).  

  It returns the maximum point value y0. 
*/           
//@{

/*!
   \ingroup resolution
   \brief     
   It returns the maximum point value y0. 
*/
template <class RandomAccessIterType>
   float parabolic_3points_fit(const RandomAccessIterType& begin_iter,
   const RandomAccessIterType& end_iter);
/*!
   \ingroup resolution
   \brief     
   It returns the maximum point location x0 in (-1,1). 
*/
template <class RandomAccessIterType>
float parabolic_3points_fit_x0(const RandomAccessIterType& begin_iter,
							   const RandomAccessIterType& end_iter);
//@}


END_NAMESPACE_STIR
