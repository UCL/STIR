//
//
/*
Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
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
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  \ingroup numerics
  \brief implementation of stir::sample_function_on_regular_grid
*/

#include "stir_experimental/numerics/more_interpolators.h"
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
    (BasicCoordinate<3,positionT>(max_out)+static_cast<positionT>(.001)) * step + offset;
  for (;
       index_out[1]<=max_out[1] && relative_positions[1]<=max_relative_positions[1];
       ++index_out[1], relative_positions[1]+= step[1])
    {
      index_out[2]=min_out[2];
      relative_positions[2]= index_out[2] * step[2] + offset[2] ;                  
      for (;
           index_out[2]<=max_out[2] && relative_positions[2]<=max_relative_positions[2];
           ++index_out[2], relative_positions[2]+= step[2])
        {
          index_out[3]=min_out[3];
          relative_positions[3]= index_out[3] * step[3] + offset[3] ;                   
          for (;
               index_out[3]<=max_out[3] && relative_positions[3]<=max_relative_positions[3];
               ++index_out[3], relative_positions[3]+= step[3])                           
            out[index_out] = func(relative_positions) ;                                                                                          
        }                        
    }                             
}

template <class elemT, class positionT>
void sample_function_on_regular_grid_pull(Array<3,elemT>& out,
                                     const Array<3,elemT>& in,
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
    (BasicCoordinate<3,positionT>(max_out)+static_cast<positionT>(.001)) * step + offset;
  for (;
       index_out[1]<=max_out[1] && relative_positions[1]<=max_relative_positions[1];
       ++index_out[1], relative_positions[1]+= step[1])
    {
      index_out[2]=min_out[2];
      relative_positions[2]= index_out[2] * step[2] + offset[2] ;
      for (;
           index_out[2]<=max_out[2] && relative_positions[2]<=max_relative_positions[2];
           ++index_out[2], relative_positions[2]+= step[2])
        {
          index_out[3]=min_out[3];
          relative_positions[3]= index_out[3] * step[3] + offset[3] ;
          for (;
               index_out[3]<=max_out[3] && relative_positions[3]<=max_relative_positions[3];
               ++index_out[3], relative_positions[3]+= step[3])
              out[index_out] = pull_linear_interpolate(in,
                                                       relative_positions);
        }
    }
}

template <class elemT>
void axial_position_boundary_conditions(Array<3,elemT>& array)
{
    // boundary contitions for the axial position

    for (int i=array[0].get_min_index(); i<= array[0].get_max_index(); ++i)
       {
         for (int j=array[0][0].get_min_index(); j<= array[0][0].get_max_index(); ++j)
          {
            const int min = array.get_min_index();
            const int max = array.get_max_index();
            array[min][i][j]=array[min+1][i][j];
            array[max][i][j]=array[max-1][i][j];
            //std::cerr<<min<<","<<max<<'\n';
          }
      }

}

template <class elemT>
void views_boundary_conditions(Array<3,elemT>& array)
{
    // boundary contitions for the axial position

    for (int i=array.get_min_index(); i<= array.get_max_index(); ++i)
       {
         for (int j=array[0][0].get_min_index(); j<= array[0][0].get_max_index(); ++j)
          {
            const int min = array[0].get_min_index();
            const int max = array[0].get_max_index();
            array[i][0][j]=100;
            array[i][max][j]=array[i][max-1][j];
            //std::cerr<<min<<","<<max<<'\n';
          }
      }
}


template <class elemT>
void tan_position_boundary_conditions(Array<3,elemT>& array)
{
    // boundary contitions for the axial position

    for (int i=array.get_min_index(); i<= array.get_max_index(); ++i)
       {
         for (int j=array[0].get_min_index(); j<= array[0].get_max_index(); ++j)
          {
            const int min = array[0][0].get_min_index();
            const int max = array[0][0].get_max_index();
            array[i][j][min]=array[i][j][min+1];
            array[i][j][max]=array[i][j][max-1];
            //std::cerr<<min<<","<<max<<'\n';
          }
      }
}

template <class elemT, class positionT>
void sample_function_on_regular_grid_push(Array<3,elemT>& out,
                                     const Array<3,elemT>& in,
                                     const BasicCoordinate<3, positionT>&  offset,
                                     const BasicCoordinate<3, positionT>& step)
{
  BasicCoordinate<3,int> min_in, max_in;
  IndexRange<3> in_range = in.get_index_range();
  if (!in_range.get_regular_range(min_in,max_in))
    warning("Output must be regular range!");

  BasicCoordinate<3, int> index_in;
  BasicCoordinate<3, positionT>  relative_positions;
  index_in[1]=min_in[1];
  relative_positions[1]= index_in[1] *step[1] - offset[1] ;
  const BasicCoordinate<3, positionT> max_relative_positions=
    (BasicCoordinate<3,positionT>(max_in)+static_cast<positionT>(.001)) * step + offset;
  for (;
       index_in[1]<=max_in[1] && relative_positions[1]<=max_relative_positions[1];
       ++index_in[1], relative_positions[1]+= step[1])
    {
      index_in[2]=min_in[2];
      relative_positions[2]= index_in[2] * step[2] + offset[2] ;
      for (;
           index_in[2]<=max_in[2] && relative_positions[2]<=max_relative_positions[2];
           ++index_in[2], relative_positions[2]+= step[2])
        {
          index_in[3]=min_in[3];
          relative_positions[3]= index_in[3] * step[3] + offset[3] ;
          for (;
               index_in[3]<=max_in[3] && relative_positions[3]<=max_relative_positions[3];
               ++index_in[3], relative_positions[3]+= step[3])
              push_transpose_linear_interpolate(out,
                            relative_positions,
                            in[index_in]);
        }
    }
            out*=(step[1]*step[2]*step[3]); //very important
}


END_NAMESPACE_STIR
