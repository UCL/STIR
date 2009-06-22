//
// $Id$
//
/*!
  \file
  \ingroup symmetries

  \brief Declaration of 2 classes: stir::SymmetryOperation and stir::TrivialSymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_recon_buildblock_SymmetryOperation_H__
#define __stir_recon_buildblock_SymmetryOperation_H__

#include "stir/common.h"


START_NAMESPACE_STIR

template <int num_dimensions, class coordT> class BasicCoordinate;
class ViewSegmentNumbers;
class ProjMatrixElemsForOneBin;
class ProjMatrixElemsForOneDensel;
class Bin;


/*!
  \ingroup buildblock
  \brief Encodes symmetry operation on image coordinates and projection
  data coordinates

  This class is mainly (only?) useful for ProjMatrix classes and their
  'users'. Together with DataSymmetriesForBins, it provides the basic 
  way to be able to write generic code without knowing which 
  particular symmetries the data have.

  Ideally, there would be no reference here to ProjMatrixElemsForOneBin,
  but we have to do this for efficiency. Overriding the virtual function
  will allow the compiler to inline the symmetry operations, resulting
  in a dramatic speed-up.

  Price to pay (aside from some tedious repetition in the derived classes): 
  the need for a
  SymmetryOperation::transform_proj_matrix_elems_for_one_bin member,
  and hence knowledge of the ProjMatrixElemsForOneBin class
  (This is the reason why the DataSymmetriesForBins* and
  SymmetryOperation* classes are in recon_buildblock.)

  See recon_buildblock/SymmetryOperations_PET_CartesianGrid.cxx for
  some more info.
*/
class SymmetryOperation
{
public:
  virtual inline ~SymmetryOperation() {}
  virtual inline bool is_trivial() const { return false;}
  virtual void 
    transform_bin_coordinates(Bin&) const = 0;
  virtual void 
    transform_view_segment_indices(ViewSegmentNumbers&) const = 0;
  virtual void
    transform_image_coordinates(BasicCoordinate<3,int>&) const = 0;
#if 0
  // would be useful at some point
  virtual void 
    transform_incremental_image_coordinates(BasicCoordinate<3,int>&) const = 0;
#endif

  virtual void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;

  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const;

};



/*!
  \ingroup symmetries
  \brief A class implementing the trivial case where the symmetry operation
  does nothing at all.
*/
class TrivialSymmetryOperation : public SymmetryOperation
{
public:
  inline bool is_trivial() const { return true;}
  inline void 
    transform_bin_coordinates(Bin& b) const {}
  inline void 
    transform_view_segment_indices(ViewSegmentNumbers& n) const {}
  inline void
    transform_image_coordinates(BasicCoordinate<3,int>& c) const {}
  inline void 
    transform_proj_matrix_elems_for_one_bin(
       ProjMatrixElemsForOneBin& lor) const {}

  virtual void 
    transform_proj_matrix_elems_for_one_densel(
      ProjMatrixElemsForOneDensel&) const {}
};


END_NAMESPACE_STIR

//#include "stir/recon_buildblock/SymmetryOperation.inl"

#endif
