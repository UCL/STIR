//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class Segment

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __Segment_H__
#define __Segment_H__


#include "ProjDataInfo.h" 
#include "shared_ptr.h"

START_NAMESPACE_TOMO
template <typename elemT> class Sinogram;
template <typename elemT> class Viewgram;

class PMessage;


/*!
  \brief An (abstract base) class for storing 3d projection data
  \ingroup buildblock

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
  inline void set_sinogram(const Sinogram<elemT>& s);
  //! set sinogram at a different axial_pos_num
  virtual void set_sinogram(const Sinogram<elemT> &s, int axial_pos_num) = 0;
  //! set data in segment according to viewgram \c v
  virtual void set_viewgram(const Viewgram<elemT>& v) = 0;

protected:
  shared_ptr<ProjDataInfo> proj_data_info_ptr;
  int segment_num;
  
  inline Segment(const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,const int s_num);
};

END_NAMESPACE_TOMO

#include "Segment.inl"

#endif


