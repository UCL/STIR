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
  \ingroup stir::motion_test

  \brief Test program for transformations with NonRigidObjectTransformationUsingBSplines

  \par Usage

  <pre>
  test_Transformations output_forward_transformation output_transpose_transformation image_to_deform forward_displacement_multicomponent
  OR
  test_Transformations output_forward_transformation output_transpose_transformation image_to_deform forward_displacement_x forward_displacement_y  forward_displacement_z

  </pre>

  \author Richard Brown
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include <iostream>

#include "stir/DiscretisedDensity.h"
#include "local/stir/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "local/stir/motion/Transform3DObjectImageProcessor.h"
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
class TransformationTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TransformationTests(const string &to_transform,
                      const string &ground_truth,
                      const string &disp_x,
                      const string &disp_y,
                      const string &disp_z,
                      const string &disp_4D);

  void run_tests();
protected:

  int _bspline_order;
  string _to_transform, _ground_truth, _disp_x, _disp_y, _disp_z, _disp_4D;

  //! run the test
  /*! Note that this function is not specific to PoissonLogLikelihoodWithLinearModelForMeanAndProjData */
  void run_test_motion_fwrd_and_back();
};

TransformationTests::
TransformationTests(const string &to_transform,
                    const string &ground_truth,
                    const string &disp_x,
                    const string &disp_y,
                    const string &disp_z,
                    const string &disp_4D) :
    _to_transform(to_transform),
    _ground_truth(ground_truth),
    _disp_x(disp_x),
    _disp_y(disp_y),
    _disp_z(disp_z),
    _disp_4D(disp_4D)
{
    if (_to_transform.size() == 0) everything_ok = false;
    if (_disp_x.size() == 0)       everything_ok = false;
    if (_disp_y.size() == 0)       everything_ok = false;
    if (_disp_z.size() == 0)       everything_ok = false;
    if (_disp_4D.size() == 0)      everything_ok = false;
    _bspline_order = 1;
}

static
void
do_fwrd_back_transformation(shared_ptr<DiscretisedDensity<3,float> > &fwrd,
                            shared_ptr<DiscretisedDensity<3,float> > &back,
                            shared_ptr<DiscretisedDensity<3,float> > &to_transform,
                            const int bspline_order,
                            const string &disp_x,
                            const string &disp_y,
                            const string &disp_z,
                            const string &disp_4D)
{
    shared_ptr<NonRigidObjectTransformationUsingBSplines<3,float> > fwrd_non_rigid;
    // 4D or 3D
    if (disp_x.size()==0)
        fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(disp_4D,bspline_order));
    else
        fwrd_non_rigid.reset(new NonRigidObjectTransformationUsingBSplines<3,float>(disp_x, disp_y, disp_z, bspline_order));

    // Image processors
    shared_ptr<Transform3DObjectImageProcessor<float> > fwrd_transform
            ( new Transform3DObjectImageProcessor<float>(fwrd_non_rigid));
    shared_ptr<Transform3DObjectImageProcessor<float> > back_transform
            ( new Transform3DObjectImageProcessor<float>(*fwrd_transform));
    back_transform->set_do_transpose(!fwrd_transform->get_do_transpose());

    std::cout << "\n\tDoing forward transformation...\n";
    fwrd.reset(to_transform->clone());
    fwrd_transform->apply(*fwrd);

    std::cout << "\n\tDoing backward transformation...\n";
    back.reset(fwrd->clone());
    back_transform->apply(*back);
}

void
TransformationTests::
run_test_motion_fwrd_and_back()
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

    shared_ptr<DiscretisedDensity<3,float> > fwrd_4D, fwrd_3D, back_4D, back_3D;

    std::cout << "\nDoing 4D transformation...\n";
    do_fwrd_back_transformation(fwrd_4D, back_4D, input, _bspline_order, "", "", "", _disp_4D);

    std::cout << "\nDoing 3D transformation...\n";
    do_fwrd_back_transformation(fwrd_3D, back_3D, input, _bspline_order, _disp_x, _disp_y, _disp_z, "");

    set_tolerance(0.1);
    check_if_equal(*ground_truth, *fwrd_4D,  "4D forward transformation does not equal ground truth");
    check_if_equal(*ground_truth, *fwrd_3D,  "3D forward transformation does not equal ground truth");

    // Some information is lost by transforming part of the image out of the FOV.
    // When we move it back, it's blank, so we can't do comparison.
    //check_if_equal(*input, *back_3D, "3D: fwrd->back should equal original input. doesn't.");
    //check_if_equal(*input, *back_4D, "4D: fwrd->back should equal original input. doesn't.");

    // Only write if ITK is present (since you want 3d and 4d in same format, but it's hassle to
    // check that the format supports both 3d and 4d writing.
#ifdef HAVE_ITK
    ITKOutputFileFormat output_file_format;
    output_file_format.default_extension = ".nii";
    output_file_format.write_to_file("STIRtmp_fwrd_4D", *fwrd_4D);
    output_file_format.write_to_file("STIRtmp_back_4D", *back_4D);
    output_file_format.write_to_file("STIRtmp_fwrd_3D", *fwrd_3D);
    output_file_format.write_to_file("STIRtmp_back_3D", *back_3D);
#endif
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

int main(int argc, char **argv)
{
    if (argc != 7) {
        cerr << "\n\tUsage: " << argv[0] << " <1> <2> <3> <4> <5> <6>\n";
        cerr << "\t\t<1>: Image to deform\n";
        cerr << "\t\t<2>: Forward transformed result (ground truth)\n";
        cerr << "\t\t<3>: Forward displacement field x component\n";
        cerr << "\t\t<4>: Forward displacement field y component\n";
        cerr << "\t\t<5>: Forward displacement field z component\n";
        cerr << "\t\t<6>: Forward displacement field image multi-component\n";
        return EXIT_SUCCESS;
    }

    set_default_num_threads();

    TransformationTests tests(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);

    if (tests.is_everything_ok())
        tests.run_tests();

    return tests.main_return_value();
}
