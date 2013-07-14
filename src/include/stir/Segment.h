/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2012 Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::Segment

  \author Kris Thielemans
  \author PARAPET project
*/
#ifndef __Segment_H__
#define __Segment_H__


#include "stir/ProjDataInfo.h" 
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR
template <typename elemT> class Sinogram;
template <typename elemT> class Viewgram;


/*!
  \brief An (abstract base) class for storing 3d projection data
  \ingroup projdata

  This stores a subset of the data accessible via a ProjData object,
  where the segment_num is fixed.

  At the moment, 2 'storage modes' are supported (and implemented as
  derived classes).

  The template argument \c elemT is used to specify the data-type of the 
  elements of the 3d object.
 */
  
template <typename elemT>
class Segment
{
#ifdef SWIG
  // need to make typedef public for swig
 public:
#endif
  typedef Segment<elemT> self_type;
public:
  
  enum StorageOrder{ StorageByView, StorageBySino };
  
  virtual ~Segment() {}
  //! Get the proj data info pointer
  inline const ProjDataInfo* get_proj_data_info_ptr() const;

  virtual StorageOrder get_storage_order() const = 0;
  //! Get the segment number
  inline int get_segment_num() const;
  virtual int get_min_axial_pos_num() const = 0;
  virtual int get_max_axial_pos_num() const = 0;
  virtual int get_min_view_num() const = 0;
  virtual int get_max_view_num() const = 0;
  virtual int get_min_tangential_pos_num()  const = 0;
  virtual int get_max_tangential_pos_num()  const = 0;  
  virtual int get_num_axial_poss() const = 0;

  virtual int get_num_views() const = 0;
  virtual int get_num_tangential_poss()  const = 0;

  //! return a new sinogram, with data set as in the segment
  virtual Sinogram<elemT> get_sinogram(int axial_pos_num) const = 0;
  //! return a new viewgram, with data set as in the segment
  virtual Viewgram<elemT> get_viewgram(int view_num) const = 0;

  //! set data in segment according to sinogram \c s
  virtual void set_sinogram(const Sinogram<elemT>& s) = 0;
  //! set sinogram at a different axial_pos_num
  virtual void set_sinogram(const Sinogram<elemT> &s, int axial_pos_num) = 0;
  //! set data in segment according to viewgram \c v
  virtual void set_viewgram(const Viewgram<elemT>& v) = 0;

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
  virtual bool operator ==(const self_type&) const = 0; 
  
  //! negation of operator==
  bool operator !=(const self_type&) const; 
  //@}

protected:
  shared_ptr<ProjDataInfo> proj_data_info_ptr;
  int segment_num;
  
  inline Segment(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,const int s_num);
};

END_NAMESPACE_STIR

#include "stir/Segment.inl"

#endif


