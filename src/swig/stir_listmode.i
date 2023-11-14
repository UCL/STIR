/*
    Copyright (C) 2023 University College London
    Copyright (C) 2022 Positrigo
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \brief Interface file for SWIG: stir::ListModeData, stir:LmToProjData, stir::ListRecord

  \author Kris Thielemans
  \author Markus Jehl
*/
%shared_ptr(stir::ListRecord);
%shared_ptr(stir::ListEvent);
%shared_ptr(stir::CListRecord);
%shared_ptr(stir::CListEvent);
%shared_ptr(stir::CListRecordWithGatingInput);

%include "stir/listmode/ListRecord.h"
%include "stir/listmode/ListEvent.h"
%include "stir/listmode/CListRecord.h"

%rename (get_empty_record) *::get_empty_record_sptr;

%rename (set_template_proj_data_info) *::set_template_proj_data_info_sptr;
%shared_ptr(stir::LmToProjData);
%include "stir/listmode/LmToProjData.h"
ADD_REPR_PARAMETER_INFO(stir::LmToProjData);

%shared_ptr(stir::ListModeData);
%include "stir/listmode/ListModeData.h"

%extend stir::ListModeData {
  static shared_ptr<stir::ListModeData> read_from_file(const std::string& filename)
    {
      using namespace stir;
      shared_ptr<ListModeData> ret(read_from_file<ListModeData>(filename));
      return ret;
    }
}

%extend stir::CListModeData {
  static shared_ptr<stir::CListModeData> read_from_file(const std::string& filename)
    {
      using namespace stir;
      shared_ptr<CListModeData> ret(read_from_file<CListModeData>(filename));
      return ret;
    }
}
