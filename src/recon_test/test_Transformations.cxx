/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
  \ingroup recon_test

  \brief Test program for stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  \par Usage

  <pre>
  test_PoissonLogLikelihoodWithLinearModelForMeanAndProjData [proj_data_filename [ density_filename ] ]
  </pre>
  where the 2 arguments are optional. See the class documentation for more info.

  \author Kris Thielemans
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include <iostream>

#include "stir/DiscretisedDensity.h"
#include "local/stir/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "local/stir/motion/Transform3DObjectImageProcessor.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/IO/ITKOutputFileFormat.h"
#include "stir/IO/OutputFileFormat.h"

START_NAMESPACE_STIR

using namespace std;

/*!
  \ingroup test
  \brief Test class for Transformations

*/
class TransformationTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TransformationTests();

  void run_tests();
protected:

  int _bspline_order;
  string _disp_field_multi_component, _disp_field_x, _disp_field_y, _disp_field_z;
  string _to_transform, _fwrd_output, _back_output;

  //! run the test
  /*! Note that this function is not specific to PoissonLogLikelihoodWithLinearModelForMeanAndProjData */
  void run_test_motion_fwrd_and_back();

public:
  /// Set inputs
  void set_inputs(string fwrd_output,
                  string back_output,
                  string to_transform,
                  string disp_field_x,
                  string disp_field_y,
                  string disp_field_z);

  /// Set inputs
  void set_inputs(string fwrd_output,
                  string back_output,
                  string to_transform,
                  string disp_field_multi_component);
};

TransformationTests::
TransformationTests()
{
    _bspline_order = 3;
}

void
TransformationTests::
set_inputs(string fwrd_output, string back_output, string to_transform, string disp_field_x, string disp_field_y, string disp_field_z)
{
    _disp_field_multi_component = "";
    _disp_field_x = disp_field_x;
    _disp_field_y = disp_field_y;
    _disp_field_z = disp_field_z;
    _to_transform = to_transform;
    _fwrd_output  = fwrd_output;
    _back_output  = back_output;
}

void
TransformationTests::
set_inputs(string fwrd_output, string back_output, string to_transform, string disp_field_multi_component)
{
    _disp_field_multi_component = disp_field_multi_component;
    _disp_field_x = "";
    _disp_field_y = "";
    _disp_field_z = "";
    _to_transform = to_transform;
    _fwrd_output  = fwrd_output;
    _back_output  = back_output;
}

void
TransformationTests::
run_test_motion_fwrd_and_back()
{
    ITKOutputFileFormat output_file_format;
    output_file_format.default_extension = ".nii";

    // Open image
    shared_ptr<DiscretisedDensity<3,float> > input;
    std::cerr << "\nabout to read the image to transform...\n";
    input.reset( DiscretisedDensity<3,float>::read_from_file(_to_transform));

    // Create non rigid transformations
    shared_ptr<NonRigidObjectTransformationUsingBSplines<3,float> > fwrd_non_rigid;
    if (_disp_field_multi_component != "") {
        fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(_disp_field_multi_component,_bspline_order));
    }
    else if (_disp_field_x != "" && _disp_field_y != "" && _disp_field_z != "")
        fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(_disp_field_x,_disp_field_y,_disp_field_z,_bspline_order));

    // Create image processors
    shared_ptr<Transform3DObjectImageProcessor<float> > fwrd_transform, back_transform;
    fwrd_transform.reset( new Transform3DObjectImageProcessor<float>(fwrd_non_rigid) );
    back_transform.reset( new Transform3DObjectImageProcessor<float>(*fwrd_transform) );
    back_transform->set_do_transpose(!fwrd_transform->get_do_transpose());

    // Start the timer
    HighResWallClockTimer t;
    t.reset();
    t.start();

    shared_ptr<DiscretisedDensity<3,float> > forward;
    forward.reset(input->clone());
    fwrd_transform->apply(*forward);
    output_file_format.write_to_file(_fwrd_output, *forward);
    cerr << "OK!\n";
/*
    cerr << "\ndoing the transpose transformation...\n";
    shared_ptr<DiscretisedDensity<3,float> > transpose;
    transpose.reset(forward->clone());
    back_transform->apply(*transpose);
    output_file_format.write_to_file(_back_output, *transpose);
    cerr << "OK!\n";
*/
    t.stop();
    cout << "Total Wall clock time: " << t.value() << " seconds" << endl;
}

void
TransformationTests::
run_tests()
{
    try {
        cerr << "Tests for TransformationTests\n";
        this->run_test_motion_fwrd_and_back();
    }
    catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
    if (argc != 5 && argc != 7) {

        cerr << "\n\tUsage: " << argv[0] << " <1> <2> <3> <4>\n";
        cerr << "\t<1>: Output of forward transformation\n";
        cerr << "\t<2>: Output of transpose transformation\n";
        cerr << "\t<3>: Image to deform\n";
        cerr << "\t<4>: Forward displacement field image multi-component\n";

        cerr << "\n\t\t\tOR\n";

        cerr << "\n\tUsage: " << argv[0] << " <1> <2> <3> <4> <5> <6>\n";
        cerr << "\t<1>: Output of forward transformation\n";
        cerr << "\t<2>: Output of transpose transformation\n";
        cerr << "\t<3>: Image to deform\n";
        cerr << "\t<4>: Forward displacement field x component\n";
        cerr << "\t<5>: Forward displacement field y component\n";
        cerr << "\t<6>: Forward displacement field z component\n";

        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TransformationTests tests;
    if (argc == 5)
        tests.set_inputs(argv[1], argv[2], argv[3], argv[4]);
    else
        tests.set_inputs(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
    tests.run_tests();

    return tests.main_return_value();
}
