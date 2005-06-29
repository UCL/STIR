//
// $Id$
//
/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  
	$Date$
	$Revision$
*/

START_NAMESPACE_STIR

/* requirement for Function3D:
   elemT Function3D::operator(const BasicCoordinate<3, positionT>&)
 */
template <class Function3D, class elemT, class positionT>
void sample_at_regular_array(Array<3,elemT>& out,
				 Function3D func,
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
				out[index_out]=
				func(relative_positions) ; 						  					  
		}			 
	}				  
}

END_NAMESPACE_STIR
