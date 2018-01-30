//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
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

  \brief Declaration of class stir::SegmentBySinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe  
  \author PARAPET project


*/
#ifndef __SegmentBySinogram_H__
#define __SegmentBySinogram_H__

#include "stir/Segment.h"
#include "stir/Array.h"
#include "stir/Sinogram.h"

START_NAMESPACE_STIR

//forward declaration for use in convertion 
template <typename elemT> class SegmentByView;


/*!
  \ingroup projdata
  \brief A class for storing (3d) projection data with a fixed segment_num.

  Storage order is as follows:
  \code
  segment_by_sino[view_num][axial_pos_num][tangential_pos_num]
  \endcode  
*/
template <typename elemT>
class SegmentBySinogram : public Segment<elemT>, public Array<3,elemT>
{
private:
  typedef SegmentBySinogram<elemT> self_type;

public:
  //! typedef such that we do not need to have \a typename wherever we StorageOrder
  typedef typename Segment<elemT>::StorageOrder StorageOrder;

  //! Constructor that sets the data to a given 3d Array
  SegmentBySinogram(const Array<3,elemT>& v, 
		    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num);
  
  //! Constructor that sets sizes via the ProjDataInfo object, initialising data to 0
  SegmentBySinogram(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num);

  
  //! Conversion from 1 storage order to the other
  SegmentBySinogram (const SegmentByView<elemT>& );
  //! Get storage order 
  inline StorageOrder get_storage_order() const;
  //! Get number of axial positions
  inline int get_num_axial_poss() const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangetial positions
  inline int get_num_tangential_poss() const;
  //! Get minimum axial position number
  inline int get_min_axial_pos_num() const;
  //! Get maximum axial position number
  inline int get_max_axial_pos_num() const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum tangetial position number
  inline int get_min_tangential_pos_num()  const;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num()  const;
  //! Get sinogram
  inline Sinogram<elemT> get_sinogram(int axial_pos_num) const;  
  //! Get viewgram
  Viewgram<elemT> get_viewgram(int view_num) const;
  //! Set viewgram
  void set_viewgram(const Viewgram<elemT>&);
  //! Set sinogram
  inline void set_sinogram(Sinogram<elemT> const &s, int axial_pos_num);  
  inline void set_sinogram(const Sinogram<elemT>& s);

  //! Overloading Array::grow
  void grow(const IndexRange<3>& range);
  //! Overloading Array::resize
  void resize(const IndexRange<3>& range);

  virtual bool operator ==(const Segment<elemT>&) const;
};

END_NAMESPACE_STIR

#include "stir/SegmentBySinogram.inl"
#endif
