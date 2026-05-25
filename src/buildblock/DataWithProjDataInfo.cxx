/*
    Copyright (C) 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief implementation of stir::DataWithProjDataInfo

  \author Kris Thielemans
*/
#include "stir/DataWithProjDataInfo.h"

START_NAMESPACE_STIR

DataWithProjDataInfo::DataWithProjDataInfo()
{}

DataWithProjDataInfo::DataWithProjDataInfo(const shared_ptr<const ProjDataInfo>& arg)
    : proj_data_info_sptr(arg)
{}

DataWithProjDataInfo::~DataWithProjDataInfo()
{}

const ProjDataInfo&
DataWithProjDataInfo::get_proj_data_info() const
{
  return *proj_data_info_sptr;
}

shared_ptr<const ProjDataInfo>
DataWithProjDataInfo::get_proj_data_info_sptr() const
{
  return proj_data_info_sptr;
}

END_NAMESPACE_STIR
