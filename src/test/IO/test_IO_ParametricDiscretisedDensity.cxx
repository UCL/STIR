//
//
/*!

  \file
  \ingroup IO

  \brief A simple program to test the stir::OutputFileFormat function.

  \author Kris Thielemans
  \author Richard Brown



  To run the test, you should use a command line argument with the name of a file.
  This should contain a test par file.
  See stir::OutputFileFormatTests class documentation for file contents.

  \warning Overwrites files STIRtmp.* in the current directory

  \todo The current implementation requires that the output file format as also
  readable by stir::read_from_file. At least we should provide a
  run-time switch to not run that part of the tests.
*/
/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
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

#include "stir/IO/test/test_IO.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the OutputFileFormat function.

  The class reads input from a stream, whose contents should be as
  follows:

  \verbatim
  Test OutputFileFormat Parameters:=
  output file format type :=
  ; here are parameters specific for the file format
  End:=
  \endverbatim

  \warning Overwrites files STIRtmp.* in the current directory
  \todo Delete STIRtmp.* files, but that's a bit difficult as we don't know which ones
  are written.
*/
class IOTests_ParametricDiscretisedDensity : public IOTests<ParametricVoxelsOnCartesianGrid>
{
public:
    explicit IOTests_ParametricDiscretisedDensity(istream& in) : IOTests(in) {}

protected:
    void create_image();
    void check_result();
};
void IOTests_ParametricDiscretisedDensity::create_image()
{
    shared_ptr<VoxelsOnCartesianGrid<float> > param_1_sptr  = create_single_image();
    shared_ptr<VoxelsOnCartesianGrid<float> > param_2_sptr  = create_single_image();
    shared_ptr<VoxelsOnCartesianGrid<float> > dummy_im_sptr = create_single_image();

    //! Setup the scanner details first
    const Scanner::Type test_scanner=Scanner::E966;
    const shared_ptr<Scanner> scanner_sptr(new Scanner(test_scanner));
    VectorWithOffset<int> num_axial_pos_per_segment; num_axial_pos_per_segment.resize(0,0);  num_axial_pos_per_segment[0]=48;
    VectorWithOffset<int> min_ring_diff;  min_ring_diff.resize(0,0);  min_ring_diff[0]=0;
    VectorWithOffset<int> max_ring_diff; max_ring_diff.resize(0,0);  max_ring_diff[0]=0;
    const int num_views=144; const int num_tangential_poss=144;

    const float zoom=1.F;

    const CartesianCoordinate3D<int> sizes (
                dummy_im_sptr->get_z_size(),
                dummy_im_sptr->get_y_size(),
                dummy_im_sptr->get_x_size());

    ProjDataInfoCylindricalNoArcCorr proj_data_info(scanner_sptr,num_axial_pos_per_segment,min_ring_diff,max_ring_diff,num_views,num_tangential_poss);

    _image_to_write_sptr.reset(new ParametricVoxelsOnCartesianGrid(ParametricVoxelsOnCartesianGridBaseType(proj_data_info,zoom,dummy_im_sptr->get_grid_spacing(),sizes)));

    // Fill the first param
    param_2_sptr->fill(2.F);
    _image_to_write_sptr->update_parametric_image(*param_1_sptr,1);

    // Fill the second param with 1's
    param_2_sptr->fill(1.F);
    _image_to_write_sptr->update_parametric_image(*param_2_sptr,2);

    // Set the time definitions
    _image_to_write_sptr->get_exam_info_sptr()->time_frame_definitions = dummy_im_sptr->get_exam_info_sptr()->time_frame_definitions;
}

void IOTests_ParametricDiscretisedDensity::check_result()
{
    set_tolerance(.00001);

    if(!check_if_equal(_image_to_read_sptr->get_num_params(),_image_to_write_sptr->get_num_params(), "test number of dynamic images"))
        return;

    // Check the exam info
    std::cerr << "\tChecking the exam info...\n";
    check_exam_info(_image_to_write_sptr->get_exam_info(),_image_to_read_sptr->get_exam_info());

    for (int i=1; i<=_image_to_read_sptr->get_num_params(); i++) {

        std::cerr << "\t\tChecking kinetic parameter " << i << "...\n";

        const VoxelsOnCartesianGrid<float> &image_to_write =
                _image_to_write_sptr->construct_single_density(i);

        const VoxelsOnCartesianGrid<float> &image_to_read =
                _image_to_read_sptr->construct_single_density(i);

        compare_images(image_to_write, image_to_read);
    }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "Usage : " << argv[0] << " filename\n"
             << "See source file for the format of this file.\n\n";
        return EXIT_FAILURE;
    }

    ifstream in(argv[1]);
    if (!in) {
        cerr << argv[0]
             << ": Error opening input file " << argv[1] << "\nExiting.\n";
        return EXIT_FAILURE;
    }

    IOTests_ParametricDiscretisedDensity tests(in);

    if (tests.is_everything_ok())
        tests.run_tests();

    return tests.main_return_value();
}
