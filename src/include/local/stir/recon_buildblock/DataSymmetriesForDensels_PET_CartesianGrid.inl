//
//
/*!
  \file
  \ingroup recon_buildblock
  \brief inline implementations for class DataSymmetriesForDensels_PET_CartesianGrid

  \author Kris Thielemans

*/
/*
    Copyright (C) 2001- 2002, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/recon_buildblock/SymmetryOperations_PET_CartesianGrid.h"

#include <algorithm>

START_NAMESPACE_STIR

#if 0
const DiscretisedDensityOnCartesianGrid<3,float> *    
DataSymmetriesForDensels_PET_CartesianGrid::
cartesian_grid_info_ptr() const
{
  // we can use static_cast here, as the constructor already checked that it is type-safe
  return static_cast<const DiscretisedDensityOnCartesianGrid<3,float> *>
    (image_info_ptr.get());
}
#endif

float
DataSymmetriesForDensels_PET_CartesianGrid::
get_num_planes_per_axial_pos(const int segment_num) const
{
  return static_cast<float>(num_planes_per_axial_pos[segment_num]);
}

float
DataSymmetriesForDensels_PET_CartesianGrid::
get_num_planes_per_scanner_ring() const
{
  return static_cast<float>(num_planes_per_scanner_ring);
}

float 
DataSymmetriesForDensels_PET_CartesianGrid::
get_axial_pos_to_z_offset(const int segment_num) const
{
  return axial_pos_to_z_offset[segment_num];
}	



SymmetryOperation* 
DataSymmetriesForDensels_PET_CartesianGrid::
find_sym_op_general_densel(const int z, const int y, const int x) const
{ 
  const int z_shift = z - (z%num_independent_planes);
  // TODO next shift might depend on the segment.
  // Solving this will require removing the axial_pos_num_shift argument from the symmetry operations
  const int axial_pos_num_shift = z_shift/num_independent_planes; 
  const int view180 = num_views;
  
  if (x>=y && y>=0) // [0,45]
    {   
      if (z_shift==0)
	return new TrivialSymmetryOperation();
      else
	return new SymmetryOperation_PET_CartesianGrid_z_shift(axial_pos_num_shift, z_shift);
    }
  else if ( x>=0 && y>x)  // [ 45,  90] 
    return new SymmetryOperation_PET_CartesianGrid_swap_xy_yx(view180, axial_pos_num_shift, z_shift);
  else if (x<0 && y>-x)   //[90, 135 ]		
    return new SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(view180, axial_pos_num_shift, z_shift);
  else if ( x<0 && y>=0)  // [ 135, 180] 
    {
      assert(y<=-x);
      return new SymmetryOperation_PET_CartesianGrid_swap_xmx(view180, axial_pos_num_shift, z_shift);
    }
  else if ( x<0 && y<=0 && -y<=-x )  // [ 180, 225] 
    return new SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(view180, axial_pos_num_shift, z_shift);
  else if ( x<=0 && y<0)  // [ 225, 270]
    {
      assert(-y>-x);
      return new SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(view180, axial_pos_num_shift, z_shift);
    }
  else if ( x>0 && -y>x)  // [ 270, 315]
    return new SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(view180, axial_pos_num_shift, z_shift);
  else // [ 315, 360]
    {
      assert(x>0 && y<0 && -y<=x);
      return new SymmetryOperation_PET_CartesianGrid_swap_ymy(view180, axial_pos_num_shift, z_shift);
    }
}


bool  
DataSymmetriesForDensels_PET_CartesianGrid::
find_basic_densel(Densel& c) const 
{
  int& z = c[1];
  int& y = c[2];
  int& x = c[3];
  if (z==z%num_independent_planes && x>=0 && y>=0 && y<=x)
    return false;
  z = z%num_independent_planes;
  if (x<0) x = -x;
  if (y<0) y = -y;
  if (y>x) std::swap(x,y);
  return true;
}


// TODO, optimise
auto_ptr<SymmetryOperation>
DataSymmetriesForDensels_PET_CartesianGrid::
  find_symmetry_operation_from_basic_densel(Densel& c) const
{
  auto_ptr<SymmetryOperation> 
    sym_op(
        find_sym_op_general_densel(c[1], c[2], c[3])
      ); 
#ifndef NDEBUG
  const Densel copy_original = c;
#endif
  find_basic_densel(c);
#ifndef NDEBUG
  Densel copy = c;
  sym_op->transform_image_coordinates(copy);
  assert(copy_original==copy);
#endif
  return sym_op;
}


int
DataSymmetriesForDensels_PET_CartesianGrid::
num_related_densels(const Densel& b) const
{
  int num = 1;
  if (b[3]!=0)
    num *= 2;
  if (b[2]!=0)
    num *= 2;
  if (abs(b[3])!=abs(b[2]))
    num *= 2;

  num *= static_cast<int>(ceil(static_cast<float>(num_planes)/num_independent_planes));
  return num;
}

void
DataSymmetriesForDensels_PET_CartesianGrid::
get_related_densels(vector<Densel>& v, const Densel& d) const
{
#ifndef NDEBUG
  {
    Densel dcopy = d;
    assert(find_basic_densel(dcopy));
  }
#endif
  v.reserve(num_related_densels(d));
  v.resize(0);
  
  const int x = d[3];
  const int y = d[2];
  const int basic_z =d[1];
  {
    for (int z=basic_z; z<num_planes; z+=num_independent_planes)
      v.push_back(Densel(z, y, x));
  }
  if (x>0)
  {
    for (int z=basic_z; z<num_planes; z+=num_independent_planes)
      v.push_back(Densel(z, y, -x));
  }
  if (y>0)
  {
    for (int z=basic_z; z<num_planes; z+=num_independent_planes)
      v.push_back(Densel(z, -y, x));
  }
  if (x>0 && y>0)
  {
    for (int z=basic_z; z<num_planes; z+=num_independent_planes)
      v.push_back(Densel(z, -y, -x));
  }
  if (x!=y)
    {
      {
	for (int z=basic_z; z<num_planes; z+=num_independent_planes)
	  v.push_back(Densel(z, x, y));
      }
      if (x>0)
	{
	  for (int z=basic_z; z<num_planes; z+=num_independent_planes)
	    v.push_back(Densel(z, x, -y));
  }
      if (y>0)
	{
	  for (int z=basic_z; z<num_planes; z+=num_independent_planes)
	    v.push_back(Densel(z, -x, y));
	}
      if (x>0 && y>0)
	{
	  for (int z=basic_z; z<num_planes; z+=num_independent_planes)
	    v.push_back(Densel(z, -x, -y));
	}
    }

  assert(v.size() == static_cast<unsigned>(num_related_densels(d)));
  
}
  
END_NAMESPACE_STIR
