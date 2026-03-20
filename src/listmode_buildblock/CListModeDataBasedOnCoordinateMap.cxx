/*
    Copyright 2015 ETH Zurich, Institute of Particle Physics
    Copyright 2020 Positrigo AG, Zurich
    Copyright 2021 University College London

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for detail..
*/

#include <iostream>:
#include <fstream>
#include "stir/Succeeded.h"

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"

START_NAMESPACE_STIR;

std::string
CListModeDataBasedOnCoordinateMap::get_name() const
{
  return listmode_filename;
}

END_NAMESPACE_STIR
