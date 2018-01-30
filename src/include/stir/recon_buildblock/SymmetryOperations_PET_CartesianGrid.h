//
//
/*!

  \file
  \ingroup symmetries
  
  \brief Declaration of all symmetry classes for PET (cylindrical) scanners and cartesian images

  \see class stir::DataSymmetriesForBins_PET_CartesianGrid
  
  \warning These classes should only be used by the 
  stir::DataSymmetriesForBins_PET_CartesianGrid class.

  \warning It is strongly recommended not to derive from any of these
  classes. If you do, you have to reimplement the 
  transform_proj_matrix_elems_for_one_bin() member, or the wrong
  implementations will be called.

  All these classes have transform_proj_matrix_elems_for_one_bin()
  members which essentially repeats just the default 
  implementation. This is for efficiency. See
  recon_buildblock/SymmetryOperations_PET_CartesianGrid.cxx for 
  more info.
    
  \author Kris Thielemans
  \author PARAPET project
      

*/          
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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


#ifndef __SymmetryOperations_PET_CartesianGrid_H__
#define __SymmetryOperations_PET_CartesianGrid_H__

#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel& ) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

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
    transform_image_coordinates(BasicCoordinate<3,int>& c) const;

  void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;


  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

private:
  int view180;
  int axial_pos_shift;
  int z_shift;
  int q;
};



END_NAMESPACE_STIR

#include "stir/recon_buildblock/SymmetryOperations_PET_CartesianGrid.inl"


#endif
