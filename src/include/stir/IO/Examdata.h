 
#ifndef __stir_IO_Examdata_H__
#define __stir_IO_Examdata_H__
/*!
  \file
  \ingroup
  \brief

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

class Examdata : public ParsingObject
{
public:

    Examdata();

    virtual ~Examdata();

protected:


private:

};

END_NAMESPACE_STIR

#endif
