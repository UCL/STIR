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

START_NAMESPACE_STIR
                          
/*!
   \ingroup resolution
   \brief find width at a level
   \param[in] begin_iterator start of sequence to check
   \param[in] max_iterator location from where the search will start, should be close 
      to the location of the (local) maximum you want to investigate
   \param[in] end_iterator end of sequence to check
   \param[in] level_height intensity level at which to find the width
   \return width of the level set (in pixel units)

   The implementation assumes uniform spacing between the samples. 

   We use linear interpolation between samples to estimate where the function value
   reaches the \a level_height value.

   If the search runs to the end/start of the sequence, a warning is issued, and the width
   is estimated using linear extrapolation.
*/
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& max_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)  ;  
/*!
   \ingroup resolution
   \brief find width at a level
   \param[in] begin_iterator start of sequence to check
   \param[in] end_iterator end of sequence to check
   \param[in] level_height intensity level at which to find the width
   \return width of the level set (in pixel units)

   This function finds the maximum in the sequence and calls 
   find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& max_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)
*/
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height);  
/*!
   \ingroup resolution
   \brief 
   finds the maximum of the Input_Array in the given slice at the given
   dimension (z=1, y=2, x=3), and returns its location as a vector in BasicCoordinate Field 
   (only 3D implementation).
*/            
template<class elemT>
BasicCoordinate<3,int>
maximum_location_per_slice(const Array<3,elemT>& input_array,
                           const int slice, const int dimension);

/*!
   \ingroup resolution
   \brief extract a line from the given array after determining its locatin with a parabolic fit

  It finds the real maximum location, using the 3 points parabolic fit. Then, tri-linear interpolation is used  
  to find the whole line at the given dimension (z:1,y:2,x:3) taking into account the voxels that
  intersects the voxel which has in its center the point with the real maximum_value.
*/ 
template <class elemT>
Array<1,elemT>
interpolate_line(const Array<3,elemT>& input_array,    
                  const BasicCoordinate<3,int>& max_location,
                  const BasicCoordinate<3,bool>& do_direction, 
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

   If nema=false, the interpolate_line() function is used to find a line, otherwise we use  extract_line().
*/

template <class elemT>           
std::list<ResolutionIndex<3,float> > 
find_fwhm_in_image(DiscretisedDensity<3,elemT> & input_image,
                   const unsigned int num_maxima, const float level, 
                   const int dimension, const bool nema);
END_NAMESPACE_STIR
