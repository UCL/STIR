//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Declaration of 2 classes: SymmetryOperation and TrivialSymmetryOperation

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __SymmetryOperation_H__
#define __SymmetryOperation_H__

#include "Tomography_common.h"


START_NAMESPACE_TOMO

template <class coordT> class Coordinate3D;
class ViewSegmentNumbers;
class ProjMatrixElemsForOneBin;
class Bin;


/*!
  \ingroup buildblock
  \brief Encodes symmetry operation on image coordinates and projection
  data coordinates

  TODOdoc

  Ideally, there would be no reference here to ProjMatrixElemsForOneBin,
  but we have to do this for efficiency. Overriding the virtual function
  will allow the compiler to inline the symmetry operations, resulting
  in a dramatic speed-up.
*/
class SymmetryOperation
{
public:
  virtual inline bool is_trivial() const { return false;}
  virtual void 
    transform_bin_coordinates(Bin&) const = 0;
  virtual void 
    transform_view_segment_indices(ViewSegmentNumbers&) const = 0;
  virtual void
    transform_image_coordinates(Coordinate3D<int>&) const = 0;
#if 0
  // would be useful at some point
  virtual void 
    transform_incremental_image_coordinates(Coordinate3D<int>&) const = 0;
#endif

  virtual void 
    transform_proj_matrix_elems_for_one_bin(
      ProjMatrixElemsForOneBin& lor) const;
#if 0
  virtual void 
    transform_proj_matrix_elems_for_voxel(
      ProjMatrixElemsForOneVoxel& out,
      const ProjMatrixElemsForOneVoxel& in) const = 0;
#endif
};



/*!
  \ingroup buildblock
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
    transform_image_coordinates(Coordinate3D<int>& c) const {}
  inline void 
    transform_proj_matrix_elems_for_one_bin(
       ProjMatrixElemsForOneBin& lor) const {}
};


END_NAMESPACE_TOMO

//#include "SymmetryOperation.inl"

#endif
