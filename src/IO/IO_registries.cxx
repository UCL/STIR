//
// $Id$
//
/*!

  \file
  \ingroup IO

  \brief File that registers all RegisterObject children in IO

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/IO/ECAT6OutputFileFormat.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT7OutputFileFormat.h"
#endif

START_NAMESPACE_STIR

static InterfileOutputFileFormat::RegisterIt dummy1;
static ECAT6OutputFileFormat::RegisterIt dummy2;
#ifdef HAVE_LLN_MATRIX
static ECAT7OutputFileFormat::RegisterIt dummy3;
#endif
END_NAMESPACE_STIR
