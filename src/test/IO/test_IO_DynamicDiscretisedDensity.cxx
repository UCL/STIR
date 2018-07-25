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
    Copyright (C) 2018- , University College London

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
#include "stir/DynamicDiscretisedDensity.h"

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
class IOTests_DynamicDiscretisedDensity : public IOTests<DynamicDiscretisedDensity>
{
public:
    explicit IOTests_DynamicDiscretisedDensity(istream& in) : IOTests(in) {}

protected:

    void create_image();
    void check_result();
};

void IOTests_DynamicDiscretisedDensity::create_image()
{
    double im_1_start = 20;
    double im_1_end   = 45;
    double im_2_start = 50;
    double im_2_end   = 93;

    shared_ptr<VoxelsOnCartesianGrid<float> > dyn_im_1_sptr = create_single_image();
    shared_ptr<VoxelsOnCartesianGrid<float> > dyn_im_2_sptr = create_single_image();
    shared_ptr<VoxelsOnCartesianGrid<float> > dummy_im_sptr = create_single_image();
    
    dyn_im_1_sptr->fill(2.);
    dyn_im_2_sptr->fill(1.);
    dyn_im_1_sptr->get_exam_info_sptr()->time_frame_definitions.set_time_frame(0,im_1_start, im_1_end);
    dyn_im_2_sptr->get_exam_info_sptr()->time_frame_definitions.set_time_frame(0,im_2_start, im_2_end);
    
    // Create a scanner (any will do)
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Advance));
    
    TimeFrameDefinitions tdefs;
    tdefs.set_num_time_frames(2);
    tdefs.set_time_frame(0,im_1_start,im_1_end);
    tdefs.set_time_frame(1,im_2_start,im_2_end);

    _image_to_write_sptr.reset(new DynamicDiscretisedDensity(tdefs,dummy_im_sptr->get_exam_info_sptr()->start_time_in_secs_since_1970,scanner_sptr,dummy_im_sptr));
    _image_to_write_sptr->set_density_sptr(dyn_im_1_sptr,1);
    _image_to_write_sptr->set_density_sptr(dyn_im_2_sptr,2);
    _image_to_write_sptr->get_exam_info_sptr()->set_high_energy_thres(dummy_im_sptr->get_exam_info_sptr()->get_high_energy_thres());
    _image_to_write_sptr->get_exam_info_sptr()->set_low_energy_thres(dummy_im_sptr->get_exam_info_sptr()->get_low_energy_thres());
}

void IOTests_DynamicDiscretisedDensity::check_result()
{
    set_tolerance(.00001);

    check_if_equal(_image_to_read_sptr->get_densities().size(),_image_to_write_sptr->get_densities().size(), "test number of dynamic images");
    
    // Check the exam info
    std::cerr << "\tChecking the exam info...\n";
    check_exam_info(_image_to_write_sptr->get_exam_info(),_image_to_read_sptr->get_exam_info());

    for (int i=1; i<=_image_to_read_sptr->get_densities().size(); i++) {

        std::cerr << "\t\tChecking dynamic image " << i << "...\n";
        
        // Cast the discretised density to voxels on cartesian grids to check grid spacing
        VoxelsOnCartesianGrid<float> *image_to_write_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *>(&_image_to_write_sptr->get_density(i));
        VoxelsOnCartesianGrid<float> *image_to_read_ptr  = dynamic_cast<VoxelsOnCartesianGrid<float> *>(&_image_to_read_sptr->get_density(i));

        if (image_to_write_ptr==0 || image_to_read_ptr==0) {
            everything_ok = false;
            return;
        }

        compare_images(*image_to_write_ptr, *image_to_read_ptr);
    }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " filename\n"
         << "See source file for the format of this file.\n\n";
    return EXIT_FAILURE;
  }

  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0]
         << ": Error opening input file " << argv[1] << "\nExiting.\n";

    return EXIT_FAILURE;
  }

  IOTests_DynamicDiscretisedDensity tests(in);
  tests.run_tests();
  return tests.main_return_value();
}
