//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London

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
  \ingroup densitydata 
  \brief Implementations of stir::VoxelsOnCartesianGrid 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project


*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/PixelsOnCartesianGrid.h"
#include "stir/NumericType.h"
#include "stir/IndexRange3D.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/utilities.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/IO/read_data.h"
#include "stir/info.h"
#include <fstream>
#include <algorithm>
#include <math.h>
#include <memory>
#include "stir/unique_ptr.h"
#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::max;
#endif
#include <boost/format.hpp>

START_NAMESPACE_STIR

// a local help function to find appropriate sizes etc.

static void find_sampling_and_z_size(
                                 float& z_sampling,
                                 float& s_sampling,
                                 int& z_size,
                                 const ProjDataInfo* proj_data_info_ptr)
{

  // first z- things

  if (const ProjDataInfoCylindrical*
        proj_data_info_cyl_ptr = 
        dynamic_cast<const ProjDataInfoCylindrical*>(proj_data_info_ptr))

   {
    // the case of cylindrical data

    z_sampling = proj_data_info_cyl_ptr->get_ring_spacing()/2;
     
    // for 'span>1' case, we take z_size = number of sinograms in segment 0
    // for 'span==1' case, we take 2*num_rings-1

    // first check if we have segment 0
    assert(proj_data_info_cyl_ptr->get_min_segment_num() <= 0);
    assert(proj_data_info_cyl_ptr->get_max_segment_num() >= 0);

    if (z_size<0)
      z_size = 
        proj_data_info_cyl_ptr->get_max_ring_difference(0) >
        proj_data_info_cyl_ptr->get_min_ring_difference(0)
        ? proj_data_info_cyl_ptr->get_num_axial_poss(0)
        : 2*proj_data_info_cyl_ptr->get_num_axial_poss(0) - 1;
  }
  else
  {
    // this is any other weird projection data. We just check sampling of segment 0
    
    // first check if we have segment 0
    assert(proj_data_info_cyl_ptr->get_min_segment_num() <= 0);
    assert(proj_data_info_cyl_ptr->get_max_segment_num() >= 0);

    // TODO make this independent on segment etc.
    z_sampling = 
      proj_data_info_ptr->get_sampling_in_t(Bin(0,0,1,0));

    if (z_size<0)
      z_size = proj_data_info_ptr->get_num_axial_poss(0);
  }

  // now do s_sampling

  {
    s_sampling =
      proj_data_info_ptr->get_scanner_ptr()->get_default_bin_size();
    if (s_sampling ==0)
      {
        // TODO make this independent on segment etc.
        s_sampling = 
          proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));
        info(boost::format("Determining voxel size from default_bin_size failed.\n"
                           "Using sampling_in_s for central bin %1%.") %
             s_sampling);
      }
    else
      {
        info(boost::format("Determined voxel size by dividing default_bin_size (%1%) by zoom") %
             s_sampling);
      }

  }

}


template<class elemT>
VoxelsOnCartesianGrid<elemT> ::VoxelsOnCartesianGrid()
 : DiscretisedDensityOnCartesianGrid<3,elemT>()
{}

template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
                      (const Array<3,elemT>& v,
                       const CartesianCoordinate3D<float>& origin,
                       const BasicCoordinate<3,float>& grid_spacing)
                       :DiscretisedDensityOnCartesianGrid<3,elemT>
                       (v.get_index_range(),origin,grid_spacing)
{
  Array<3,elemT>::operator=(v);
}


template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
                      (const IndexRange<3>& range, 
                       const CartesianCoordinate3D<float>& origin,
                       const BasicCoordinate<3,float>& grid_spacing)
                       :DiscretisedDensityOnCartesianGrid<3,elemT>
                       (range,origin,grid_spacing)
{}

template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
                      (const shared_ptr < ExamInfo > & exam_info_sptr,
                       const Array<3,elemT>& v,
                       const CartesianCoordinate3D<float>& origin,
                       const BasicCoordinate<3,float>& grid_spacing)
                       :DiscretisedDensityOnCartesianGrid<3,elemT>
                       (exam_info_sptr,v.get_index_range(),origin,grid_spacing)
{
  Array<3,elemT>::operator=(v);
}


template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
                      (const shared_ptr < ExamInfo > & exam_info_sptr,
                       const IndexRange<3>& range, 
                       const CartesianCoordinate3D<float>& origin,
                       const BasicCoordinate<3,float>& grid_spacing)
                       :DiscretisedDensityOnCartesianGrid<3,elemT>
                        (exam_info_sptr,range,origin,grid_spacing)
{}

// KT 10/12/2001 use new format of args for the constructor, and remove the make_xy_size_odd constructor
template<class elemT>                                                 
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid(const ProjDataInfo& proj_data_info,
                                                    const float zoom, 
                                                    const CartesianCoordinate3D<float>& origin,
                                                    const CartesianCoordinate3D<int>& sizes)
                                                    
{
  init_from_proj_data_info(proj_data_info, zoom, origin, sizes);
}

template<class elemT>                                                 
VoxelsOnCartesianGrid<elemT>::
VoxelsOnCartesianGrid(const shared_ptr<ExamInfo>& exam_info_sptr_v,
                      const ProjDataInfo& proj_data_info,
                      const float zoom,
                      const CartesianCoordinate3D<float>& origin,
                      const CartesianCoordinate3D<int>& sizes)
{
  this->exam_info_sptr = exam_info_sptr_v;
  init_from_proj_data_info(proj_data_info, zoom, origin, sizes);
}

template<class elemT>
void
VoxelsOnCartesianGrid<elemT>::
init_from_proj_data_info(const ProjDataInfo& proj_data_info,
                         const float zoom,
                         const CartesianCoordinate3D<float>& origin,
                         const CartesianCoordinate3D<int>& sizes)
{
  int z_size = sizes.z();
  // initialise to 0 to prevent compiler warnings
  //int z_size = 0;
  float z_sampling = 0;
  float s_sampling = 0;
  find_sampling_and_z_size(z_sampling, s_sampling, z_size, &proj_data_info);
  
  this->set_grid_spacing(
      CartesianCoordinate3D<float>(z_sampling, s_sampling/zoom, s_sampling/zoom)
      );
  int x_size_used = sizes.x();
  int y_size_used = sizes.y();

  if (sizes.x()==-1 || sizes.y()==-1)
    {
      // default it to cover full FOV by taking image_size>=2*FOVradius_in_pixs+1
      const float FOVradius_in_mm = 
        max(proj_data_info.get_s(Bin(0,0,0,proj_data_info.get_max_tangential_pos_num())),
            -proj_data_info.get_s(Bin(0,0,0,proj_data_info.get_min_tangential_pos_num())));
      if (sizes.x()==-1)
        x_size_used = 2*static_cast<int>(ceil(FOVradius_in_mm / get_voxel_size().x())) + 1;
      if (sizes.y()==-1)
        y_size_used = 2*static_cast<int>(ceil(FOVradius_in_mm / get_voxel_size().y())) + 1;        
    }
  if (x_size_used<0)
    error("VoxelsOnCartesianGrid: attempt to construct image with negative x_size %d\n", 
          x_size_used);
  if (x_size_used==0)
    warning("VoxelsOnCartesianGrid: constructed image with x_size 0\n");
  if (y_size_used<0)
    error("VoxelsOnCartesianGrid: attempt to construct image with negative y_size %d\n", 
          y_size_used);
  if (y_size_used==0)
    warning("VoxelsOnCartesianGrid: constructed image with y_size 0\n");

  IndexRange3D range (0, z_size-1, 
                      -(y_size_used/2), -(y_size_used/2) + y_size_used-1,
                      -(x_size_used/2), -(x_size_used/2) + x_size_used-1);

  this->set_origin(origin);
  this->set_vendor_origin
    (origin
     + (proj_data_info
        .get_location_of_vendor_frame_of_reference_in_physical_coordinates()));

  this->grow(range);
}

/*!
  This member function will be unnecessary when all compilers can handle
  'covariant' return types. 
  It is a non-virtual counterpart of get_empty_voxels_on_cartesian_grid.
*/
template<class elemT>
VoxelsOnCartesianGrid<elemT>*
VoxelsOnCartesianGrid<elemT>::get_empty_voxels_on_cartesian_grid() const

{
  return new VoxelsOnCartesianGrid(this->get_exam_info_sptr()->create_shared_clone(),
                                   this->get_index_range(),
                                   this->get_origin(), 
                                   this->get_grid_spacing());
}


template<class elemT>
#ifdef STIR_NO_COVARIANT_RETURN_TYPES
DiscretisedDensity<3,elemT>*
#else
VoxelsOnCartesianGrid<elemT>*
#endif
VoxelsOnCartesianGrid<elemT>::get_empty_copy() const
{
  return get_empty_voxels_on_cartesian_grid();
}

template<class elemT>
#ifdef STIR_NO_COVARIANT_RETURN_TYPES
DiscretisedDensity<3,elemT>*
#else
VoxelsOnCartesianGrid<elemT>*
#endif
VoxelsOnCartesianGrid<elemT>::clone() const
{
  VoxelsOnCartesianGrid *temp = new VoxelsOnCartesianGrid(*this);
  temp->set_exam_info(*temp->get_exam_info_sptr()->create_shared_clone());
  return temp;
}

template<class elemT>
void 
VoxelsOnCartesianGrid<elemT>::set_voxel_size(const BasicCoordinate<3,float>& c) 
{
  this->set_grid_spacing(c);
}

template<class elemT>  
PixelsOnCartesianGrid<elemT>                                          
VoxelsOnCartesianGrid<elemT>::get_plane(const int z) const
{
  PixelsOnCartesianGrid<elemT> 
    plane(this->operator[](z),
          this->get_origin(),
          Coordinate2D<float>(get_voxel_size().y(), get_voxel_size().x())
          );
  return plane;
}

/*! This function requires that the dimensions, origin and grid_spacings match. */
template<class elemT>   
void                                          
VoxelsOnCartesianGrid<elemT>::set_plane(const PixelsOnCartesianGrid<elemT>& plane, const int z)
{
  assert(this->get_min_x() == plane.get_min_x());
  assert(this->get_max_x() == plane.get_max_x());
  assert(this->get_min_y() == plane.get_min_y());
  assert(this->get_max_y() == plane.get_max_y());
  assert(this->get_origin() == plane.get_origin());
  assert(this->get_voxel_size().x() == plane.get_pixel_size().x());
  assert(this->get_voxel_size().y() == plane.get_pixel_size().y());
  
  this->operator[](z) = plane;    
}

template<class elemT>   
void                                          
VoxelsOnCartesianGrid<elemT>::grow_z_range(const int min_z, const int max_z)
{
  /* This is somewhat complicated as Array is not very good with regular ranges.
     It works by 
     - getting the regular range, 
     - 'grow' this by hand, 
     - make a general IndexRange from this
     - call Array::grow with the general range
  */
  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;

  this->get_regular_range(min_indices, max_indices);
  assert(min_z <= min_indices.z());
  assert(max_z >= max_indices.z());
  min_indices.z() = min_z;
  max_indices.z() = max_z;
  this->grow(IndexRange<3>(min_indices, max_indices));
}

template<class elemT>
BasicCoordinate<3,int>
VoxelsOnCartesianGrid<elemT>::
get_lengths() const
{
  return make_coordinate(this->get_z_size(), this->get_y_size(), this->get_x_size());
}

template<class elemT>
BasicCoordinate<3,int>
VoxelsOnCartesianGrid<elemT>::
get_min_indices() const
{
  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;
  this->get_regular_range(min_indices, max_indices);
  return min_indices;
}

template<class elemT>
BasicCoordinate<3,int>
VoxelsOnCartesianGrid<elemT>::
get_max_indices() const
{
  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;
  this->get_regular_range(min_indices, max_indices);
  return max_indices;
}
#if 0

/****************************
 static members
 ***************************/
template<class elemT>
VoxelsOnCartesianGrid<elemT> VoxelsOnCartesianGrid<elemT>::ask_parameters()
{
  // this is completely superseded by read_from_file
  // TODO make into something else useful?

  // Open file with data
  ifstream input;
  
  ask_filename_and_open(
    input, "Enter filename for input image", ".v", 
    ios::in | ios::binary);

  unique_ptr<Scanner> scanner_ptr
    (Scanner::ask_parameters());


  NumericType data_type;
  {
    int data_type_sel = ask_num("Type of data :\n\
0: signed 16bit int, 1: unsigned 16bit int, 2: 4bit float ", 0,2,2);
    switch (data_type_sel)
      { 
      case 0:
        data_type = NumericType::SHORT;
        break;
      case 1:
        data_type = NumericType::USHORT;
        break;
      case 2:
        data_type = NumericType::FLOAT;
        break;
      }
  }


  {
    // find offset 

    input.seekg(0L, ios::beg);   
    unsigned long file_size = find_remaining_size(input);

    unsigned long offset_in_file = ask_num("Offset in file (in bytes)", 
                             0UL,file_size, 0UL);
    input.seekg(offset_in_file, ios::beg);
  }

  
  CartesianCoordinate3D<float> 
    origin(0,0,0);
  CartesianCoordinate3D<float>
    voxel_size(scanner_ptr->get_ring_spacing()/2,
               scanner_ptr->get_default_bin_size(),
               scanner_ptr->get_default_bin_size()); 
  int num_bins_from_scanner = scanner_ptr->get_default_num_arccorrected_bins();
 int num_rings_from_scanner = scanner_ptr->get_num_rings();
  int max_bin = (- num_bins_from_scanner /2) +  num_bins_from_scanner -1;
  if (num_bins_from_scanner % 2 == 0 &&
      ask("Make x,y size odd ?", true))
    max_bin++;
   

  VoxelsOnCartesianGrid<elemT> 
    input_image(IndexRange3D(
                                0, 2*num_rings_from_scanner-2,
                                (-num_bins_from_scanner/2), max_bin,
                                (-num_bins_from_scanner/2), max_bin),
                origin,
                voxel_size);


  // TODO handle scale factor in case of not reading float
  float scale = float(1);
  // note: currently stir:: needed to avoid conflict with Array::read_data
  stir::read_data(input, input_image,
            data_type, scale);  
  assert(scale==1);

  return input_image; 

}
#endif

/**********************************************
 instantiations
 **********************************************/
template class VoxelsOnCartesianGrid<float>;
template class VoxelsOnCartesianGrid<CartesianCoordinate3D<float> >;

END_NAMESPACE_STIR

#include "stir/modelling/KineticParameters.h"
namespace stir {
  template class VoxelsOnCartesianGrid<KineticParameters<1,float> >; 
  template class VoxelsOnCartesianGrid<KineticParameters<2,float> >; 
  template class VoxelsOnCartesianGrid<KineticParameters<3,float> >; 
}
