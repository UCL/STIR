//
//
/*!
  \file
  \ingroup projection
  
  \brief Declaration of class stir::ProjMatrixElemsForOneDenselValue
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_ProjMatrixElemsForOneDenselValue_H__
#define __stir_recon_buildblock_ProjMatrixElemsForOneDenselValue_H__

#include "stir/Bin.h"

START_NAMESPACE_STIR


/*!
  \ingroup projection
  \brief Stores voxel coordinates and the value of the matrix element. 
 
  (Probably) only useful in class ProjMatrixElemsForOneDensel.

  \warning It is recommended never to use this class name directly, but
  always use the typedef ProjMatrixElemsForOneDensel::value_type.


  \todo Simply derived from Bin for now.

 */
 class ProjMatrixElemsForOneDenselValue : public Bin
{ 
public:
  explicit inline
    ProjMatrixElemsForOneDenselValue(const Bin&);

  inline ProjMatrixElemsForOneDenselValue();

  //! Adds el2.get_value() to the value of the current object
  inline ProjMatrixElemsForOneDenselValue& operator+=(const ProjMatrixElemsForOneDenselValue& el2);
  //! Multiplies the value of with a float
  inline ProjMatrixElemsForOneDenselValue& operator*=(const float d);
  //! Adds a float to the value 
  inline ProjMatrixElemsForOneDenselValue& operator+=(const float d);
  //! Divides the value of with a float
  inline ProjMatrixElemsForOneDenselValue& operator/=(const float d);

  // TODO
  inline float get_value() const { return get_bin_value(); }
  inline void set_value(const float v)  { set_bin_value(v); }
  
  //////// comparison functions

  //! Checks if the coordinates are equal
  /*! This function and the next one below are implemented as static members,
      such that you can pass them (as function objects) to std::sort.
   */
  static inline bool coordinates_equal(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2);

  //! Checks lexicographical order of the coordinates
  static inline bool coordinates_less(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2);
   
  //! Checks coordinates and value are equal
  friend inline bool operator==(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2);
  
  //! Checks lexicographical order of the coordinates and the value
  friend inline bool operator<(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2);
 
  
};


END_NAMESPACE_STIR

#include "stir/recon_buildblock/ProjMatrixElemsForOneDenselValue.inl"

#endif // __ProjMatrixElemsForOneDenselValue_H__
