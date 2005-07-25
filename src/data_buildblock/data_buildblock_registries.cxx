//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup data_buildblock

  \brief File that registers all stir::RegisterObject children in data_buildblock

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
#include "stir/common.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/data/SinglesRatesFromECAT7.h"
#include "stir/data/SinglesRatesFromSglFile.h"
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

END_NAMESPACE_STIR

