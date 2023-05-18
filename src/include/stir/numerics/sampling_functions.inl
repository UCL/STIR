//
//
/*
Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
Copyright 2023, Positrigo AG, Zurich
This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0
        
          See STIR/LICENSE.txt for details
*/
/*!
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  \author Markus Jehl
  \ingroup numerics
  \brief implementation of stir::sample_function_on_regular_grid and
         stir::sample_function_using_index_converter
*/
#include "stir/warning.h"
START_NAMESPACE_STIR

template <class FunctionType, class elemT, class positionT>
void sample_function_on_regular_grid(Array<3,elemT>& out,
                                     FunctionType func,
                                     const BasicCoordinate<3, positionT>&  offset,  
                                     const BasicCoordinate<3, positionT>& step)
{
  BasicCoordinate<3,int> min_out, max_out;
  IndexRange<3> out_range = out.get_index_range(); 
  if (!out_range.get_regular_range(min_out,max_out))
    warning("Output must be regular range!");
                
  BasicCoordinate<3, int> index_out;
  BasicCoordinate<3, positionT>  relative_positions;
  index_out[1]=min_out[1];
  relative_positions[1]= index_out[1] * step[1] - offset[1] ;
  const BasicCoordinate<3, positionT> max_relative_positions= 
    (BasicCoordinate<3,positionT>(max_out)+static_cast<positionT>(.001)) * step - offset;
  for (;
       index_out[1]<=max_out[1] && relative_positions[1]<=max_relative_positions[1];
       ++index_out[1], relative_positions[1]+= step[1])
    {
      index_out[2]=min_out[2];
      relative_positions[2]= index_out[2] * step[2] - offset[2] ;                  
      for (;
           index_out[2]<=max_out[2] && relative_positions[2]<=max_relative_positions[2];
           ++index_out[2], relative_positions[2]+= step[2])
        {
          index_out[3]=min_out[3];
          relative_positions[3]= index_out[3] * step[3] - offset[3] ;                   
          for (;
               index_out[3]<=max_out[3] && relative_positions[3]<=max_relative_positions[3];
               ++index_out[3], relative_positions[3]+= step[3])                           
            out[index_out] = func(relative_positions) ;                                                                                          
        }                        
    }                             
}

template <typename elemT, typename FunctionType, typename Lambda>
void sample_function_using_index_converter(Array<3,elemT>& out, FunctionType func, Lambda&& index_converter)
{
  BasicCoordinate<3,int> min_out, max_out;
  IndexRange<3> out_range = out.get_index_range();
  if (!out_range.get_regular_range(min_out,max_out))
    warning("Output must be regular range!");

  BasicCoordinate<3, int> index_out;
  for (index_out[1] = min_out[1]; index_out[1] <= max_out[1]; ++index_out[1])
  {
    for (index_out[2] = min_out[2]; index_out[2] <= max_out[2]; ++index_out[2])
    {
      for (index_out[3] = min_out[3]; index_out[3] <= max_out[3]; ++index_out[3])
      {
        auto index_in = index_converter(index_out);
        out[index_out] = func(index_in);
      }
    }
  } 
}

END_NAMESPACE_STIR
