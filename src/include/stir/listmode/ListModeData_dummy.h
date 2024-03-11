/*
    Copyright (C) 2022, MGH / HST A. Martinos Center for Biomedical Imaging
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::ListModeData_dummy

  \author Nikos Efthimiou
*/

#ifndef __stir_listmode_ListModeData_dummy_H__
#define __stir_listmode_ListModeData_dummy_H__

#include "stir/error.h"

START_NAMESPACE_STIR
/*!
  \brief A class to trick the Objective function that we have list mode data,
  when we only have cache.

  \ingroup listmode
  \author Nikos Efthimiou

 This class was created only to hold an ExamInfo and be returned
 by ListModeData& get_input_data() during set_up() when calling
 STIR from SIRF.

 This can happen when a private (not open-source) IO is required to read
 the listmode files. However, the owner of that library shared a cache
 (see PoissonLogLikelihoodWithLinearModelForMeanAndListModeData)
listmode file with a third party.


 \todo Currently, it can only be used in combination with SIRF.
 \todo A possibility is to complete this as a class for cache files.

!*/
class ListModeData_dummy : public ListModeData
{
public:
  ListModeData_dummy(shared_ptr<const ExamInfo> exam_info, shared_ptr<const ProjDataInfo> proj_data_info)
  {
    this->exam_info_sptr = exam_info;
    this->proj_data_info_sptr = proj_data_info;
  }

  std::string get_name() const override { return std::string("ListModeData_dummy"); }

  Succeeded reset() override
  {
    error("ListModeData_dummy does not have reset()");
    return Succeeded::no;
  }

  SavedPosition save_get_position() override
  {
    error("ListModeData_dummy does not have save_get_position()");
    return 0;
  }

  Succeeded set_get_position(const SavedPosition&) override
  {
    error("ListModeData_dummy does not have set_get_position()");
    return Succeeded::no;
  }

  bool has_delayeds() const override
  {
    error("ListModeData_dummy does not have has_delayeds()");
    return false;
  }

protected:
  shared_ptr<ListRecord> get_empty_record_helper_sptr() const override
  {
    error("ListModeData_dummy does not have get_empty_record_helper_sptr()");
    return nullptr;
  }
  Succeeded get_next(ListRecord& event) const override
  {
    error("ListModeData_dummy does not have get_next()");
    return Succeeded::no;
  }
};

END_NAMESPACE_STIR

#endif
