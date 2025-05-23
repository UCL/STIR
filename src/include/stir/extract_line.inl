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
  \ingroup buildblock
  \brief Implementation of stir::extract_line

  \author Charalampos Tsoumpas
  \author Kris Thielemans


 */

START_NAMESPACE_STIR
template <class elemT>
Array<1,elemT>
extract_line(const Array<3,elemT>& input_array,    
             const BasicCoordinate<3,int>& index, 
             const int dimension)   
{ 
  BasicCoordinate<3,int> min_index,max_index;
  min_index[1] = input_array.get_min_index();
  max_index[1] = input_array.get_max_index();
  min_index[2] = input_array[index[1]].get_min_index();
  max_index[2] = input_array[index[1]].get_max_index();
  min_index[3] = input_array[index[1]][index[2]].get_min_index();
  max_index[3] = input_array[index[1]][index[2]].get_max_index();       
  Array<1,elemT> line(min_index[dimension],max_index[dimension]);    
  BasicCoordinate<3,int> running_index = index;
  int &counter = running_index[dimension];  
  for (counter=min_index[dimension]; counter<= max_index[dimension] ; ++counter)
    line[counter]= input_array[running_index];
  return line ;
}  
                          
END_NAMESPACE_STIR
