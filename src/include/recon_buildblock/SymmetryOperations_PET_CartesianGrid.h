//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  
  \brief Declaration of all symmetry classes for PET scanners and cartesian images
    
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$

  \warning These classes should onlyy be used by the DataSymmetriesForBins_PET_CartesianGrid
  and DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid classes.
*/          

// TODOdoc
#ifndef __SymmetryOperations_PET_CartesianGrid_H__
#define __SymmetryOperations_PET_CartesianGrid_H__

#include "SymmetryOperation.h"

START_NAMESPACE_TOMO

class SymmetryOperation_PET_CartesianGrid_z_shift : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_z_shift self;
public:
  SymmetryOperation_PET_CartesianGrid_z_shift(const int axial_pos_shift, const int z_shift)
    : axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int axial_pos_shift;
  int z_shift;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmx_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmx_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmx_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};


///////////////////////////////////////

class SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmy_yx_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};


class SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xy_yx_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmy_yx : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmy_yx self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmy_yx(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};


class SymmetryOperation_PET_CartesianGrid_swap_xy_yx : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xy_yx self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xy_yx(const int num_views, const int axial_pos_shift, const int z_shift)    
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};



class SymmetryOperation_PET_CartesianGrid_swap_xmx : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmx self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmx(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};

class SymmetryOperation_PET_CartesianGrid_swap_ymy : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_ymy self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_ymy(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  };

class SymmetryOperation_PET_CartesianGrid_swap_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};

class SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xy_ymx_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};

class SymmetryOperation_PET_CartesianGrid_swap_xy_ymx : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xy_ymx self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xy_ymx(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};

class SymmetryOperation_PET_CartesianGrid_swap_ymy_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_ymy_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_ymy_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmx_ymy(const int num_views, const int axial_pos_shift, const int z_shift)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
};

class SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq : public SymmetryOperation
{
private:
  typedef SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq self;
public:
  SymmetryOperation_PET_CartesianGrid_swap_xmy_ymx_zq(const int num_views, const int axial_pos_shift, const int z_shift, const int q)
    : view180(num_views), axial_pos_shift(axial_pos_shift), z_shift(z_shift), q(q)
  {}

  inline void 
    transform_bin_coordinates(Bin&) const;
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers&) const;
  inline void
    transform_image_coordinates(Coordinate3D<int>& c) const;
  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};



END_NAMESPACE_TOMO

#include "SymmetryOperations_PET_CartesianGrid.inl"

#endif
