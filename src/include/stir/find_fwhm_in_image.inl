//
//
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup resolution
  \brief A collection of functions to measure resolution

  \author Charalampos Tsoumpas
  \author Kris Thielemans


 */
#include <algorithm>

START_NAMESPACE_STIR
                          
              
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& current_max_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)
{ 
  const int max_position = static_cast<int>(current_max_iterator - begin_iterator + 1);
  RandomAccessIterType current_iter = current_max_iterator;
  while(current_iter!= end_iterator && *current_iter > level_height)   ++current_iter;
  if (current_iter==end_iterator)  
    {
      warning("find_level_width: level extends beyond border."
              "Cannot find the real level-width of this point source!");
      // go 1 back to remain inside the range
      --current_iter;
    }

  // do linear interpolation to find position of level_height  
  float right_level_max = (*current_iter - level_height)/(*current_iter-*(current_iter-1));
  right_level_max = float(current_iter-(begin_iterator+max_position)) - right_level_max ;
  
  current_iter = current_max_iterator;
  while(current_iter!=begin_iterator && *current_iter > level_height) --current_iter;
  if (current_iter == begin_iterator && *current_iter > level_height) 
    {
      warning("find_level_width: level extends beyond border."
              "Cannot find the real level-width of this point source!");
    }
        
  float left_level_max = (*current_iter - level_height)/(*current_iter-*(current_iter+1));
  left_level_max += float(current_iter-(begin_iterator+max_position));

  return right_level_max - left_level_max;   
} 
                     
template <class RandomAccessIterType>
float find_level_width(const RandomAccessIterType& begin_iterator,
                       const RandomAccessIterType& end_iterator,
                       const float level_height)
{
  return find_level_width(begin_iterator, 
                          std::max_element(begin_iterator,end_iterator),
                          end_iterator,
                          level_height); 
}
END_NAMESPACE_STIR
