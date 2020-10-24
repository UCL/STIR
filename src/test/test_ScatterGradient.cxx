//
//
/*
    Copyright (C) 2019, University of Hull
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
  \ingroup test

  \brief Test program for ScatterGradient

  \author Ludovica Brusaferri

*/

#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/scatter/SingleScatterLikelihoodAndGradient.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"
#include "stir/IO/write_to_file.h"
#include <iostream>
#include <math.h>
#include "stir/centre_of_gravity.h"

using std::cerr;
using std::endl;
using std::string;

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for Scanner
*/
class ScatterGradientTests: public RunTests
{
public:  
    void run_tests();
private:
    void test_scatter_gradient();

};



void
ScatterGradientTests::test_scatter_gradient()
{
    //TODO
}

void
ScatterGradientTests::
run_tests()
{

    test_scatter_gradient();
}


END_NAMESPACE_STIR


int main()
{
    USING_NAMESPACE_STIR

    ScatterGradientTests tests;
    tests.run_tests();
    return tests.main_return_value();
}
