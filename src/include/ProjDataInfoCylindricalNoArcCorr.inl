//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline functions of class 
  ProjDataInfoCylindricalNoArcCorr

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "Bin.h"
#include <math.h>

START_NAMESPACE_TOMO

float
ProjDataInfoCylindricalNoArcCorr::get_s(const Bin& bin) const
{
  return ring_radius * sin(bin.tangential_pos_num()*angular_increment);
}



END_NAMESPACE_TOMO

