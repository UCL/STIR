//
// $Id$
//
/*!

  \file

  \brief Declaration of SSRB

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SSRB_H__
#define __stir_SSRB_H__

#include "stir/common.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class ProjData;

void 
SSRB(const string& output_filename,
     ProjData& in_projdata,
     const int span,
     const int max_segment_num_to_process,
     const bool do_norm
     );

void 
SSRB(ProjData& out_projdata,
     const ProjData& in_projdata,
     const bool do_norm
     );

END_NAMESPACE_STIR

#endif
