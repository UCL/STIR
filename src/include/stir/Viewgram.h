//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \ingroup projdata

  \brief Declaration of class stir::Viewgram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
#ifndef __Viewgram_h__
#define __Viewgram_h__


#include "stir/Array.h"
#include "stir/ProjDataInfo.h" 
#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"


START_NAMESPACE_STIR

/*!
  \ingroup projdata
  \brief A class for 2d projection data.

  This represents a subset of the full projection. segment_num and tangential_pos_num 
  are fixed.
  
*/

template <typename elemT>
class Viewgram : public Array<2,elemT>
{
private:
  typedef Array<2,elemT> base_type;
#ifdef SWIG
  // SWIG needs the next typedef to be public
public:  
#endif
  typedef Viewgram<elemT> self_type;
#ifdef SWIG
  // SWIG needs a default constructor
  inline Viewgram() {}
#endif

public:
  //! Construct from proj_data_info pointer, view and segment number. Data are set to 0.
  inline Viewgram(const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
                  const int v_num, const int s_num); 

  //! Construct with data set to the array.
  inline Viewgram(const Array<2,elemT>& p,const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
                  const int v_num, const int s_num); 
  

  //! Get segment number
  inline int get_segment_num() const; 
  //! Get number of views
  inline int get_view_num() const;
  //! Get minimum number of axial positions
  inline int get_min_axial_pos_num() const;
  //! Get maximum number of axial positions 
  inline int get_max_axial_pos_num() const;
  //! Get number of axial positions
  inline int get_num_axial_poss() const;
  //! Get minimum number of axial positions
  inline int get_min_tangential_pos_num() const;
  //! Get maximum number of tangetial positions
  inline int get_max_tangential_pos_num() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  
  //! Get an empty viewgram of the same dimensions, segment_num etc.
  inline  Viewgram get_empty_copy(void) const;

  //! Overloading Array::grow
  void grow(const IndexRange<2>& range);
  //! Overloading Array::resize
  void resize(const IndexRange<2>& range);

  //! Get the proj_data_info pointer
  /*! \warning Do not use this pointer after the Viewgram object is destructed.
  */
  inline const ProjDataInfo* get_proj_data_info_ptr() const;
 
  //! \name Equality
  //@{
  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! If they do \c not have the same characteristics, the string \a explanation
      explains why.
  */
  bool
    has_same_characteristics(self_type const&,
			     string& explanation) const;

  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! Use this version if you do not need to know why they do not match.
   */
  bool
    has_same_characteristics(self_type const&) const;

  //! check equality (data has to be identical)
  /*! Uses has_same_characteristics() and Array::operator==.
      \warning This function uses \c ==, which might not be what you 
      need to check when \c elemT has data with float or double numbers.
  */
  bool operator ==(const self_type&) const; 
  
  //! negation of operator==
  bool operator !=(const self_type&) const; 
  //@}
  
private:
  
  shared_ptr<ProjDataInfo> proj_data_info_ptr; 
  int view_num;
  int segment_num;  
};

END_NAMESPACE_STIR

#include "stir/Viewgram.inl"

#endif
