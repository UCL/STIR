//
//
/*
    Copyright (C) 2003- 2005, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup numerics
  \brief Functions to interpolate data

  \author Kris Thielemans

*/
#include "stir/Coordinate3D.h"
#include "stir/Array.h"
#include "stir/round.h"
#include <cmath>


START_NAMESPACE_STIR

template <class elemT, class positionT>
elemT
pull_nearest_neighbour_interpolate(const Array<3, elemT>& in, 
			const BasicCoordinate<3, positionT>& point_in_input_coords) 
{
  // find nearest neighbour 
  const Coordinate3D<int> 
    nearest_neighbour = round(point_in_input_coords);

  if (nearest_neighbour[1] <= in.get_max_index() &&
      nearest_neighbour[1] >= in.get_min_index() &&
      nearest_neighbour[2] <= in[nearest_neighbour[1]].get_max_index() &&
      nearest_neighbour[2] >= in[nearest_neighbour[1]].get_min_index() &&
      nearest_neighbour[3] <= in[nearest_neighbour[1]][nearest_neighbour[2]].get_max_index() &&
      nearest_neighbour[3] >= in[nearest_neighbour[1]][nearest_neighbour[2]].get_min_index())
    {
      return in[nearest_neighbour];
    }
  else
    return 0;
}

template <int num_dimensions, class elemT, class positionT, class valueT>
void
push_nearest_neighbour_interpolate(Array<num_dimensions, elemT>& out, 
				   const BasicCoordinate<num_dimensions, positionT>& point_in_output_coords,
				   valueT value)
{
  if (value==0)
   return;
  const BasicCoordinate<num_dimensions, int> nearest_neighbour =
    round(point_in_output_coords);
  
  if (nearest_neighbour[1] <= out.get_max_index() &&
      nearest_neighbour[1] >= out.get_min_index() &&
      nearest_neighbour[2] <= out[nearest_neighbour[1]].get_max_index() &&
      nearest_neighbour[2] >= out[nearest_neighbour[1]].get_min_index() &&
      nearest_neighbour[3] <= out[nearest_neighbour[1]][nearest_neighbour[2]].get_max_index() &&
      nearest_neighbour[3] >= out[nearest_neighbour[1]][nearest_neighbour[2]].get_min_index())
    out[nearest_neighbour] +=
      static_cast<elemT>(value);
}



template <class elemT, class positionT>
elemT
pull_linear_interpolate(const Array<3, elemT>& in, 
			const BasicCoordinate<3, positionT>& point_in_input_coords) 
{
  // find left neighbour 
  const Coordinate3D<int> 
  left_neighbour(round(std::floor(point_in_input_coords[1])),
		 round(std::floor(point_in_input_coords[2])),
		 round(std::floor(point_in_input_coords[3])));

  // TODO handle boundary conditions
  if (left_neighbour[1] < in.get_max_index() &&
      left_neighbour[1] > in.get_min_index() &&
      left_neighbour[2] < in[left_neighbour[1]].get_max_index() &&
      left_neighbour[2] > in[left_neighbour[1]].get_min_index() &&
      left_neighbour[3] < in[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
      left_neighbour[3] > in[left_neighbour[1]][left_neighbour[2]].get_min_index())
    {
      const int x1=left_neighbour[3];
      const int y1=left_neighbour[2];
      const int z1=left_neighbour[1];
      const int x2=left_neighbour[3]+1;
      const int y2=left_neighbour[2]+1;
      const int z2=left_neighbour[1]+1;
      const positionT ix = point_in_input_coords[3]-x1;
      const positionT iy = point_in_input_coords[2]-y1;
      const positionT iz = point_in_input_coords[1]-z1;
      const positionT ixc = 1 - ix;
      const positionT iyc = 1 - iy;
      const positionT izc = 1 - iz;
      return
	static_cast<elemT>
	( 
	 ixc * (iyc * (izc * in[z1][y1][x1]
		       + iz  * in[z2][y1][x1])
		+ iy * (izc * in[z1][y2][x1]
			+ iz  * in[z2][y2][x1])) 
	 + ix * (iyc * (izc * in[z1][y1][x2]
			+ iz  * in[z2][y1][x2])
		 + iy * (izc * in[z1][y2][x2]
			 + iz  * in[z2][y2][x2]))
	 );
    }
  else
    return 0;
}

template <class elemT, class positionT, class valueT>
void
push_transpose_linear_interpolate(Array<3, elemT>& out, 
				  const BasicCoordinate<3, positionT>& point_in_output_coords,
				  valueT value)
{
  if (value==0)
   return;
  // find left neighbour
  const Coordinate3D<int> 
    left_neighbour(round(std::floor(point_in_output_coords[1])),
		   round(std::floor(point_in_output_coords[2])),
		   round(std::floor(point_in_output_coords[3])));

  // TODO handle boundary conditions
  if (left_neighbour[1] < out.get_max_index() &&
      left_neighbour[1] > out.get_min_index() &&
      left_neighbour[2] < out[left_neighbour[1]].get_max_index() &&
      left_neighbour[2] > out[left_neighbour[1]].get_min_index() &&
      left_neighbour[3] < out[left_neighbour[1]][left_neighbour[2]].get_max_index() &&
      left_neighbour[3] > out[left_neighbour[1]][left_neighbour[2]].get_min_index())
    {
      const int x1=left_neighbour[3];
      const int y1=left_neighbour[2];
      const int z1=left_neighbour[1];
      const int x2=left_neighbour[3]+1;
      const int y2=left_neighbour[2]+1;
      const int z2=left_neighbour[1]+1;
      const float ix = point_in_output_coords[3]-x1;
      const float iy = point_in_output_coords[2]-y1;
      const float iz = point_in_output_coords[1]-z1;
      const float ixc = 1 - ix;
      const float iyc = 1 - iy;
      const float izc = 1 - iz;
      out[z1][y1][x1] += ixc * iyc * izc * value;
      out[z2][y1][x1] += ixc * iyc * iz  * value;
      out[z1][y2][x1] += ixc * iy * izc * value;
      out[z2][y2][x1] += ixc * iy * iz  * value;
      out[z1][y1][x2] += ix * iyc * izc * value;
      out[z2][y1][x2] += ix * iyc * iz  * value;
      out[z1][y2][x2] += ix * iy * izc * value;
      out[z2][y2][x2] += ix * iy * iz  * value;
    }
}

END_NAMESPACE_STIR
