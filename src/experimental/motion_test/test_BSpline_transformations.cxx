/*
    Copyright (C) 2018, University College London
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
  \ingroup stir::motion_test

  \brief Test program for non-rigid B-spline transformations with stir::NonRigidObjectTransformationUsingBSplines

  \par Usage

  <pre>
  test_Transformations image_to_transform ground_truth multicomponent_displacement

  </pre>

  \author Richard Brown
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include <iostream>

#include "stir/DiscretisedDensity.h"
#include "stir_experimental/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "stir_experimental/motion/Transform3DObjectImageProcessor.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/is_null_ptr.h"
#ifdef HAVE_ITK
#include "stir/IO/ITKOutputFileFormat.h"
#endif

START_NAMESPACE_STIR

using namespace std;

/*!
  \ingroup test
  \brief Test class for Transformations

*/
class TestBSplineTransformation : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TestBSplineTransformation(const string &to_transform,
                      const string &ground_truth,
                      const string &disp_4D);

  void run_tests();
protected:

  int _bspline_order;
  string _to_transform, _ground_truth, _disp_4D;

  //! run the test
  void run_transformation();
};

TestBSplineTransformation::
TestBSplineTransformation(const string &to_transform,
                    const string &ground_truth,
                    const string &disp_4D) :
    _to_transform(to_transform),
    _ground_truth(ground_truth),
    _disp_4D(disp_4D)
{
    if (_to_transform.empty()) everything_ok = false;
    if (_disp_4D.empty())      everything_ok = false;
    _bspline_order = 1;
}

void
TestBSplineTransformation::
run_transformation()
{
    // Open image
    std::cerr << "\nabout to read the image to transform...\n";
    shared_ptr<DiscretisedDensity<3,float> > input(
                DiscretisedDensity<3,float>::read_from_file(_to_transform));
    if(is_null_ptr(input)) {
        std::cerr << "\nError reading image to transform.\n";
        everything_ok = false;
        return;
    }

    // Open ground truth
    std::cerr << "\nabout to read the ground truth image...\n";
    shared_ptr<DiscretisedDensity<3,float> > ground_truth(
                DiscretisedDensity<3,float>::read_from_file(_ground_truth));
    if(is_null_ptr(ground_truth)) {
        std::cerr << "\nError reading ground truth image.\n";
        everything_ok = false;
        return;
    }

    shared_ptr<NonRigidObjectTransformationUsingBSplines<3,float> > fwrd_non_rigid(
                new NonRigidObjectTransformationUsingBSplines<3,float>(_disp_4D,_bspline_order));

    // Image processors
    shared_ptr<Transform3DObjectImageProcessor<float> > forward_transform( new Transform3DObjectImageProcessor<float>(fwrd_non_rigid));
    shared_ptr<Transform3DObjectImageProcessor<float> > adjoint_transform( new Transform3DObjectImageProcessor<float>(*forward_transform));
    adjoint_transform->set_do_transpose(!forward_transform->get_do_transpose());

    std::cout << "\n\tDoing forward transformation...\n";
    shared_ptr<DiscretisedDensity<3,float> > forward(input->clone());
    forward_transform->apply(*forward);

    std::cout << "\n\tDoing adjoint transformation...\n";
    shared_ptr<DiscretisedDensity<3,float> > adjoint(forward->clone());
    adjoint_transform->apply(*adjoint);

    // Might be slightly different due to interpolation differences
    set_tolerance(1);
    check_if_equal(*ground_truth, *forward,  "4D forward transformation does not equal ground truth");

    // Some information is lost by transforming part of the image out of the FOV.
    // When we move it back, it's blank, so we can't do following comparison.
    //check_if_equal(*input, *back_4D, "4D: fwrd->back should equal original input. doesn't.");

    // Only write if ITK is present (since you want 3d and 4d in same format, but it's hassle to
    // check that the format supports both 3d and 4d writing.
#ifdef HAVE_ITK
    ITKOutputFileFormat output_file_format;
    output_file_format.default_extension = ".nii";
    output_file_format.write_to_file("STIRtmp_forward", *forward);
    output_file_format.write_to_file("STIRtmp_adjoint", *adjoint);
#endif
}

void
TestBSplineTransformation::
run_tests()
{
    try {
        cerr << "Tests for TransformationTests\n";
        this->run_transformation();
    }
    catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
    if (argc != 4) {
        cerr << "\n\tUsage: " << argv[0] << " <1> <2> <3>\n";
        cerr << "\t\t<1>: Image to transform\n";
        cerr << "\t\t<2>: Transformed result (ground truth)\n";
        cerr << "\t\t<3>: Multi-component displacement field image\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestBSplineTransformation tests(argv[1], argv[2], argv[3]);

    if (tests.is_everything_ok())
        tests.run_tests();

    return tests.main_return_value();
}
