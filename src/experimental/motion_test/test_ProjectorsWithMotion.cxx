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

  \brief Test program for forward and backwards projectors using non-rigid B-spline transformations with stir::PresmoothingForwardProjectorByBin and stir::PostsmoothingBackProjectorByBin

  \par Usage

  <pre>
  test_ProjectorsWithMotion image_to_transform template_proj_data multicomponent_displacement projector_parameters

  The following is an example of a projector parameter file:
  Projector parameters :=
  projector pair type := Matrix
    Projector Pair Using Matrix Parameters :=
    Matrix type := Ray Tracing
    number of rays in tangential direction to trace for each bin := 10
  End Ray Tracing Matrix Parameters:=
  End Projector Pair Using Matrix Parameters :=
  End :=

  </pre>

  \author Richard Brown
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include <iostream>
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#include "stir/DiscretisedDensity.h"
#include "stir_experimental/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "stir_experimental/motion/Transform3DObjectImageProcessor.h"
#include "stir/ProjData.h"
#include "stir/is_null_ptr.h"
#ifdef HAVE_ITK
#include "stir/IO/ITKOutputFileFormat.h"
#endif

START_NAMESPACE_STIR

using namespace std;

/*!
  \ingroup test
  \brief Test class for projectors with motion

*/
class TestProjectorsWithMotion : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TestProjectorsWithMotion(const string &to_transform,
                           const string &proj_data,
                           const string &disp_4D,
                           const string &projector_parameters);

  void run_tests();
protected:

  int _bspline_order;
  string _to_transform, _disp_4D, _projector_parameters, _proj_data;

  //! run the test
  void run_transformation();
};

TestProjectorsWithMotion::
TestProjectorsWithMotion(const string &to_transform,
                         const string &proj_data,
                         const string &disp_4D,
                         const string &projector_parameters) :
    _to_transform(to_transform),
    _proj_data(proj_data),
    _disp_4D(disp_4D),
    _projector_parameters(projector_parameters)
{ _bspline_order = 1; }

void
TestProjectorsWithMotion::
run_transformation()
{
    // Open image
    std::cerr << "\nReading the image to transform...\n";
    shared_ptr<DiscretisedDensity<3,float> > input(
                DiscretisedDensity<3,float>::read_from_file(_to_transform));
    if(is_null_ptr(input)) {
        std::cerr << "\nError reading image to transform.\n";
        everything_ok = false;
        return;
    }
    std::cerr << "\tDone!\n";

    // Open sinogram
    std::cerr << "\nReading the example proj data...\n";
    shared_ptr<ProjData> proj_data =
      ProjData::read_from_file(_proj_data);
    std::cerr << "\tDone!\n";

    // Image processors
    std::cerr << "\nSetting up image processors...\n";
    shared_ptr<NonRigidObjectTransformationUsingBSplines<3,float> > fwrd_non_rigid(
                new NonRigidObjectTransformationUsingBSplines<3,float>(_disp_4D,_bspline_order));
    shared_ptr<Transform3DObjectImageProcessor<float> > forward_transform( new Transform3DObjectImageProcessor<float>(fwrd_non_rigid));
    shared_ptr<Transform3DObjectImageProcessor<float> > adjoint_transform( new Transform3DObjectImageProcessor<float>(*forward_transform));
    adjoint_transform->set_do_transpose(!forward_transform->get_do_transpose());
    std::cerr << "\tDone!\n";

    // Standard projector pair
    std::cerr << "\nSetting up standard projector pair...\n";
    shared_ptr<ProjectorByBinPair> projector_pair_sptr;
    KeyParser parser;
    parser.add_start_key("Projector Parameters");
    parser.add_parsing_key("Projector pair type", &projector_pair_sptr);
    parser.add_stop_key("END");
    if (!parser.parse(_projector_parameters.c_str()) || is_null_ptr(projector_pair_sptr))
        throw std::runtime_error("Failed to parse projector pair type (" + _projector_parameters + ").");
    std::cerr << "\tDone!\n";

    // Create projectors
    std::cerr << "\nCreating motion projectors...\n";
    shared_ptr<PresmoothingForwardProjectorByBin> fwrd_projector;
    shared_ptr<PostsmoothingBackProjectorByBin>   back_projector;
    fwrd_projector.reset(
                new PresmoothingForwardProjectorByBin(
                    projector_pair_sptr->get_forward_projector_sptr(),
                    forward_transform));
    back_projector.reset(
                new PostsmoothingBackProjectorByBin(
                    projector_pair_sptr->get_back_projector_sptr(),
                    adjoint_transform));
    std::cerr << "\tDone!\n";

    // Set up projectors
    std::cerr << "\nSetting up motion projectors...\n";
    fwrd_projector->set_input(input);
    fwrd_projector->set_up(proj_data->get_proj_data_info_sptr(), input);
    back_projector->set_up(proj_data->get_proj_data_info_sptr(), input);
    std::cerr << "\tDone!\n";

    std::cerr << "\nDoing forward projection...\n";
    //proj_data->fill(0.F);
    fwrd_projector->forward_project(*proj_data, *input);
    std::cerr << "\tDone!\n";

    shared_ptr<DiscretisedDensity<3,float> > back_projected(input->clone());

    std::cerr << "\nDoing back projection...\n";
    back_projector->start_accumulating_in_new_image();
    back_projector->back_project(*back_projected, *proj_data);
    std::cerr << "\tDone!\n";

    // Might be slightly different due to interpolation differences
    set_tolerance(1);
    check_if_equal(*input, *back_projected,  "Forward/back projected image should be same as original, but it isn't.");

    // Only write if ITK is present (since you want 3d and 4d in same format, but it's hassle to
    // check that the format supports both 3d and 4d writing.
#ifdef HAVE_ITK
    ITKOutputFileFormat output_file_format;
    output_file_format.default_extension = ".nii";
    output_file_format.write_to_file("STIRtmp_back_projected", *back_projected);
#endif
}

void
TestProjectorsWithMotion::
run_tests()
{
    try {
        cerr << "Tests for projectors with motion\n";
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
    if (argc != 5) {
        cerr << "\n\tUsage: " << argv[0] << " <1> <2> <3> <4>\n";
        cerr << "\t\t<1>: Image to transform\n";
        cerr << "\t\t<2>: Example sinogram\n";
        cerr << "\t\t<3>: Multi-component displacement field image\n";
        cerr << "\t\t<4>: Projector parameter file\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestProjectorsWithMotion tests(argv[1], argv[2], argv[3], argv[4]);

    if (tests.is_everything_ok())
        tests.run_tests();

    return tests.main_return_value();
}
