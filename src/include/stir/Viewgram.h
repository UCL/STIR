//
// $Id$
//
/*!

  \file
  \ingroup projdata

  \brief Declaration of class Viewgram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __Viewgram_h__
#define __Viewgram_h__


#include "stir/Array.h"
#include "stir/ProjDataInfo.h" 
#include "stir/IndexRange.h"
#include "stir/shared_ptr.h"


START_NAMESPACE_STIR

class PMessage;

/*!
  \ingroup projdata
  \brief A class for 2d projection data.

  This represents a subset of the full projection. segment_num and tangential_pos_num 
  are fixed.
  
*/

template <typename elemT>
class Viewgram : public Array<2,elemT>
{
  
public:
  //! Construct from proj_data_info pointer, view and segment number. Data are set to 0.
  inline Viewgram(const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
                  const int v_num, const int s_num); 

  //! Construct with data set to the array.
  inline Viewgram(const Array<2,elemT>& p,const shared_ptr<ProjDataInfo>& proj_data_info_ptr, 
                  const int v_num, const int s_num); 
  

  Viewgram(PMessage& msg);
  
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

  //! Get the proj_data_info pointer
  /*! \warning Do not use this pointer after the Viewgram object is destructed.
  */
  inline const ProjDataInfo* get_proj_data_info_ptr() const;
 
  
private:
  
  shared_ptr<ProjDataInfo> proj_data_info_ptr; 
  int view_num;
  int segment_num;  
};

END_NAMESPACE_STIR

#include "stir/Viewgram.inl"

#endif
