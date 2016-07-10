 
#ifndef __stir_IO_Study_H__
#define __stir_IO_Study_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Definition of class stir::General_Reconstruction

  \author Nikos Efthimiou
*/

#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/ParsingObject.h"
#include <vector>
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR

class Succeeded;

class Study : public ParsingObject
{
public:

    Study();

    virtual ~Study();

protected:


private:

};

END_NAMESPACE_STIR

#endif
