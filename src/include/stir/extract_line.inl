//
//
/*
    Copyright (C) 2004- 2009, Hammersmith Imanet
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
