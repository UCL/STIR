/* CListModeDataSAFIR.cxx

Coincidence LM Data Class for SAFIR: Implementation

        Copyright 2015 ETH Zurich, Institute of Particle Physics
        Copyright 2020 Positrigo AG, Zurich
    Copyright 2021 University College London

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
*/
/*!

  \file
  \ingroup listmode
  \brief implementation of class stir::CListModeDataSAFIR

  \author Jannis Fischer
  \author Kris Thielemans
  \author Markus Jehl
*/
#include <iostream>
#include <fstream>
#include "stir/Succeeded.h"

#include "stir/listmode/CListModeDataBasedOnCoordinateMap.h"

using std::ios;
using std::fstream;
using std::ifstream;
using std::istream;

START_NAMESPACE_STIR;

std::string
CListModeDataBasedOnCoordinateMap::get_name() const
{
  return listmode_filename;
}

END_NAMESPACE_STIR
