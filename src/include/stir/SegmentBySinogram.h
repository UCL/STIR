//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
  \brief A class for storing (3d) projection data with fixed SegmentIndices.

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
		    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v,
		    const SegmentIndices& ind);

  //! Constructor that sets sizes via the ProjDataInfo object, initialising data to 0
  SegmentBySinogram(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v,
		    const SegmentIndices& ind);

  //! Constructor that sets the data to a given 3d Array
  /*!
    \deprecated Use version with SegmentIndices instead
  */
  SegmentBySinogram(const Array<3,elemT>& v, 
            const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num, const int timing_pos_num = 0);
  
  //! Constructor that sets sizes via the ProjDataInfo object, initialising data to 0
  /*!
    \deprecated Use version with SegmentIndices instead
  */
  SegmentBySinogram(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v,
		    const int segment_num, const int timing_pos_num = 0);

  
  //! Conversion from 1 storage order to the other
  SegmentBySinogram (const SegmentByView<elemT>& );
  //! Get storage order 
  inline StorageOrder get_storage_order() const override;
  //! Get number of axial positions
  inline int get_num_axial_poss() const override;
  //! Get number of views
  inline int get_num_views() const override;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const override;
  //! Get minimum axial position number
  inline int get_min_axial_pos_num() const override;
  //! Get maximum axial position number
  inline int get_max_axial_pos_num() const override;
  //! Get minimum view number
  inline int get_min_view_num() const override;
  //! Get maximum view number
  inline int get_max_view_num() const override;
  //! Get minimum tangetial position number
  inline int get_min_tangential_pos_num()  const override;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num()  const override;
  using Segment<elemT>::get_sinogram;
  using Segment<elemT>::get_viewgram;
  //! Get sinogram
  inline Sinogram<elemT> get_sinogram(int axial_pos_num) const override;
  //! Get viewgram
  Viewgram<elemT> get_viewgram(int view_num) const override;
  //! Set viewgram
  void set_viewgram(const Viewgram<elemT>&) override;
  //! Set sinogram
  inline void set_sinogram(Sinogram<elemT> const &s, int axial_pos_num) override;
  inline void set_sinogram(const Sinogram<elemT>& s) override;

  //! Overloading Array::grow
  void grow(const IndexRange<3>& range) override;
  //! Overloading Array::resize
  void resize(const IndexRange<3>& range) override;

  bool operator ==(const Segment<elemT>&) const override;
};

END_NAMESPACE_STIR

#include "stir/SegmentBySinogram.inl"
#endif
