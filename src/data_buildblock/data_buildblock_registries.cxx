//
//
/*
    Copyright (C) 2005- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup data_buildblock

  \brief File that registers all stir::RegisterObject children in data_buildblock

  \author Kris Thielemans
  
*/
#include "stir/common.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/data/SinglesRatesFromECAT7.h"
#include "stir/data/SinglesRatesFromSglFile.h"
#endif
#ifdef HAVE_HDF5
#include "stir/data/SinglesRatesFromGEHDF5.h"
#endif
START_NAMESPACE_STIR
#ifdef HAVE_LLN_MATRIX
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7
static SinglesRatesFromECAT7::RegisterIt dummy100;
static SinglesRatesFromSglFile::RegisterIt dummy200;
END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
#endif
#ifdef HAVE_HDF5
static GE::RDF_HDF5::SinglesRatesFromGEHDF5::RegisterIt dummy300;
#endif

END_NAMESPACE_STIR

