//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
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
/*!
  \file
  \ingroup resolution
  \brief A collection of functions to measure resolution

  \author Charalampos Tsoumpas
  \author Kris Thielemans

  $Date$
  $Revision$

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
   \brief find width at a level
   takes as input the Begin and the End Iterators of a sequence 
   of numbers (e.g. vector) and the level you want to have (e.g. maximum/2 or maximum/10). 
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
  extracts a line from the given array (that includes the particular index)
  in the direction of the specified dimension.
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
   \brief Finds FWHM, FWTM etc (in mm) for a number of point sources or a line source   
  
   \param[in] input_image
   \param[in] num_maxima  the number of maxima to find (see below)
   \param[in] level level at which to compute the width (2 for half maximum, 10 for tenth maximum etc.)
   \param[in] dimension the dimension along which the line source is oriented, or 0 for point sources
   \param[in] nema enables the calculation based on the NEMA Standards Publication 
   NU 2-2001. 
   \return a list containing the maximum_value, its location per slice and resolution. 
   For line sources, sorted by minimum to maximum slice of the requested dimension, or for point sources, 
   sorted by maximum to minimum value. 

   For line sources, \a num_maxima slices are sampled (from first to last slice, with steps given by num_slices/(num_maxima+1)

   For point sources, if \a num_maxima is larger than 1, after finding a maximum and the resolution, the data is masked out in
   a neigbhourhood of half-size (resolution*2/level). This will only ork properly if the point sources are not too close to eachother
   and have roughly the maximum.

   The value of the maximum is computed using a parabolic fit through the 3 points around the maximum as specified
   in NEMA 2001. 

   If nema=false, the interpolated_line() function is used to find a line, otherwise we use  extract_line().
*/

template <int num_dimensions, class elemT>           
std::list<ResolutionIndex<num_dimensions,float> > 
find_fwhm_in_image(DiscretisedDensity<num_dimensions,elemT> & input_image,
                   const unsigned int num_maxima, const float level, 
                   const int dimension, const bool nema);
/*!  
   \ingroup resolution
   \brief assign a value to a sub-region of an array

   sets all values for indices between \a mask_location - \a half_size and \a mask_location + \a half_size to \a value,
   taking care of staing inside the index-range of the array.
*/
template <int num_dimensions, class elemT>   
void 
assign_to_subregion(Array<num_dimensions,elemT>& input_array, 
                    const BasicCoordinate<num_dimensions,int>& mask_location,
                    const BasicCoordinate<num_dimensions,int>& half_size,
                    const elemT& value);


END_NAMESPACE_STIR
