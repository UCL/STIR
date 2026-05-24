/*
    Copyright (C) 2000, Hammersmith Imanet Ltd
    Copyright (C) 2016, 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_IO_DataWithProjDataInfo_H__
#define __stir_IO_DataWithProjDataInfo_H__
/*!
  \file
  \ingroup buildblock
  \brief declaration of stir::DataWithProjDataInfo

  \author Kris Thielemans
*/

#include "stir/shared_ptr.h"
#include "stir/ProjDataInfo.h"

START_NAMESPACE_STIR

/*!
  \brief base class for data objects such as ProjData etc
  \ingroup buildblock

  Provides a ProjDataInfo member.
*/
class DataWithProjDataInfo
{
public:
  //!
  //! \brief Default constructor sets internal member to 0
  DataWithProjDataInfo();

  DataWithProjDataInfo(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr_v);

  virtual ~DataWithProjDataInfo();
  virtual const ProjDataInfo& get_proj_data_info() const;
  //! Get shared pointer to ProjData info
  virtual shared_ptr<const ProjDataInfo> get_proj_data_info_sptr() const;

  //! Get number of segments
  inline int get_num_segments() const;
  //! Get number of axial positions per segment
  inline int get_num_axial_poss(const int segment_num) const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  //! Get number of TOF positions
  inline int get_num_tof_poss() const;
  //! Get the index of the first timing position
  inline int get_min_tof_pos_num() const;
  //! Get the index of the last timing position.
  inline int get_max_tof_pos_num() const;
  //! Get TOG mash factor
  inline int get_tof_mash_factor() const;
  //! Get minimum segment number
  inline int get_min_segment_num() const;
  //! Get maximum segment number
  inline int get_max_segment_num() const;
  //! Get mininum axial position per segmnet
  inline int get_min_axial_pos_num(const int segment_num) const;
  //! Get maximum axial position per segment
  inline int get_max_axial_pos_num(const int segment_num) const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum tangential position number
  inline int get_min_tangential_pos_num() const;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num() const;

protected:
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;

private:
};

END_NAMESPACE_STIR

#include "stir/DataWithProjDataInfo.inl"
#endif
