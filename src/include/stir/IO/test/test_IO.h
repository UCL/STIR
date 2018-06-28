//
//
/*!

  \file
  \ingroup test

  \brief A simple program to test the stir::OutputFileFormat function.

  \author Kris Thielemans
  \author Richard Brown


  
  To run the test, you should use a command line argument with the name of a file.
  This should contain a test par file.
  See stir::IOTests class documentation for file contents.

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

#include "stir/RunTests.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT6OutputFileFormat.h" // need this for test on pixel_size
#endif
/*
#include "stir/KeyParser.h"

#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"

#include <iostream>
#include <memory>
#include <math.h>
*/
#include <fstream>

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

template <class A>
class IOTests : public RunTests
{
public:
    IOTests(std::istream& in);

    void run_tests();
protected:

    void         set_up();
    void         create_single_image();
    virtual void create_image() = 0;
    virtual void write_image();
    virtual void read_image();
    virtual void check_result() = 0;
    void         compare_images(const VoxelsOnCartesianGrid<float> &im_1,
                                const VoxelsOnCartesianGrid<float> &im_2);

    std::istream&                               _in;
    shared_ptr<OutputFileFormat<A> >            _output_file_format_sptr;
    KeyParser                                   _parser;
    std::string                                 _filename;
    shared_ptr<VoxelsOnCartesianGrid<float> >   _single_image_sptr;
    shared_ptr<A>                               _image_to_write_sptr;
    shared_ptr<A>                               _image_to_read_sptr;
};

template <class A>
IOTests<A>::
IOTests(std::istream& in) :
    _in(in)
{ }

template <class A>
void IOTests<A>::set_up()
{
    _filename = "STIRtmp";

    _output_file_format_sptr.reset();
    _parser.add_start_key("Test OutputFileFormat Parameters");
    _parser.add_parsing_key("output file format type", &_output_file_format_sptr);
    _parser.add_stop_key("END");

    std::cerr << "Testing OutputFileFormat parsing function..." << std::endl;
    std::cerr << "WARNING: will overwite files called STIRtmp*\n";

    if (!check(_parser.parse(_in), "parsing failed"))
        return;

    if (!check(_output_file_format_sptr,
               "parsing failed to set _output_file_format_sptr"))
        return;
}

template <class A>
void IOTests<A>::create_single_image()
{
    // construct density image
#ifdef HAVE_LLN_MATRIX
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT6
    // TODO get next info from OutputFileFormat class instead of hard-wiring
    // this in here
    const bool supports_different_xy_pixel_sizes =
        dynamic_cast<ECAT6OutputFileFormat const * const>(_output_file_format_sptr.get()) == 0
            ? true : false;
    const bool supports_origin_z_shift =
        dynamic_cast<ECAT6OutputFileFormat const * const>(_output_file_format_sptr.get()) == 0
            ? true : false;
    const bool supports_origin_xy_shift = true;
#else
    const bool supports_different_xy_pixel_sizes = true;
    const bool supports_origin_z_shift = true;
    const bool supports_origin_xy_shift = true;
#endif

    CartesianCoordinate3D<float> origin (0.F,0.F,0.F);
    if (supports_origin_xy_shift)
        {  origin.x()=2.4F; origin.y() = -3.5F; }
    if (supports_origin_z_shift)
        {  origin.z()=6.4F; }

    CartesianCoordinate3D<float> grid_spacing (3.F,4.F,supports_different_xy_pixel_sizes?5.F:4.F);

    IndexRange<3>
        range(CartesianCoordinate3D<int>(0,-15,-14),
        CartesianCoordinate3D<int>(4,14,14));

    VoxelsOnCartesianGrid<float> single_image(range,origin, grid_spacing);

    // fill with some data
    for (int z=single_image.get_min_z(); z<=single_image.get_max_z(); ++z)
        for (int y=single_image.get_min_y(); y<=single_image.get_max_y(); ++y)
            for (int x=single_image.get_min_x(); x<=single_image.get_max_x(); ++x)
                single_image[z][y][x]=
                    3*sin(static_cast<float>(x*_PI)/single_image.get_max_x())
                    *sin(static_cast<float>(y+10*_PI)/single_image.get_max_y())
                    *cos(static_cast<float>(z*_PI/3)/single_image.get_max_z());

    _single_image_sptr.reset(new VoxelsOnCartesianGrid<float>(single_image));
}

template <class A>
void IOTests<A>::write_image()
{
    // write to file
    const Succeeded success = _output_file_format_sptr->write_to_file(_filename,*_image_to_write_sptr);

    check(success == Succeeded::yes, "failed writing");
}

template <class A>
void IOTests<A>::read_image()
{
    // now read it back
    _image_to_read_sptr = read_from_file<A>(_filename);

    check(!is_null_ptr(_image_to_read_sptr), "failed reading");
}

template <class A>
void IOTests<A>::compare_images(const VoxelsOnCartesianGrid<float> &im_1,
                                const VoxelsOnCartesianGrid<float> &im_2)
{
    set_tolerance(.000001);

    check_if_equal(im_1.get_grid_spacing(), im_2.get_grid_spacing(), "test on read and written file via image grid spacing ");

    if (_output_file_format_sptr->get_type_of_numbers().integer_type()) {
        set_tolerance(10.*im_1.find_max()/
                  pow(2.,static_cast<double>(_output_file_format_sptr->get_type_of_numbers().size_in_bits())));
    }

    check_if_equal(im_1.get_voxel_size(), im_2.get_voxel_size(), "test on read and written file via image voxel size");
    check_if_equal(im_1.get_length(), im_2.get_length(), "test on read and written file via image length");
    check_if_equal(im_1.get_lengths(), im_2.get_lengths(), "test on read and written file via image lengths");
    check_if_equal(im_1.get_max_index(), im_2.get_max_index(), "test on read and written file via image max index");
    check_if_equal(im_1.get_max_indices(), im_2.get_max_indices(), "test on read and written file via image max indices");
    check_if_equal(im_1.get_max_x(), im_2.get_max_x(), "test on read and written file via image max x");
    check_if_equal(im_1.get_max_y(), im_2.get_max_y(), "test on read and written file via image max y");
    check_if_equal(im_1.get_max_z(), im_2.get_max_z(), "test on read and written file via image max z");
    check_if_equal(im_1.get_min_index(), im_2.get_min_index(), "test on read and written file via image min index");
    check_if_equal(im_1.get_min_indices(), im_2.get_min_indices(), "test on read and written file via image min indices");
    check_if_equal(im_1.get_min_x(), im_2.get_min_x(), "test on read and written file via image min x");
    check_if_equal(im_1.get_min_y(), im_2.get_min_y(), "test on read and written file via image min y");
    check_if_equal(im_1.get_min_z(), im_2.get_min_z(), "test on read and written file via image min z");
    check_if_equal(im_1.get_x_size(), im_2.get_x_size(), "test on read and written file via image x size");
    check_if_equal(im_1.get_y_size(), im_2.get_y_size(), "test on read and written file via image y size");
    check_if_equal(im_1.get_z_size(), im_2.get_z_size(), "test on read and written file via image z size");
    check_if_equal(im_1.find_max(), im_2.find_max(), "test on read and written file via image max");
    check_if_equal(im_1.find_min(), im_2.find_min(), "test on read and written file via image min");

    check_if_equal(im_1, im_2, "test on read and written file");

    set_tolerance(.00001);

    check_if_equal(im_1.get_origin(), im_2.get_origin(), "test on read and written file via image origin");

}

template <class A>
void IOTests<A>::run_tests()
{
    try {
        this->set_up();
        if (!everything_ok) return;

        this->create_single_image();
        if (!everything_ok) return;

        this->create_image();
        if (!everything_ok) return;

        std::cerr << "\nAbout to write the image to disk...\n";
        this->write_image();
        if (!everything_ok) return;
        std::cerr << "OK!\n";

        std::cerr << "\nAbout to read the image back from disk...\n";
        this->read_image();
        if (!everything_ok) return;
        std::cerr << "OK!\n";

        std::cerr << "\nAbout to check the consistency between the two images...\n";
        this->check_result();
        if (!everything_ok) return;
        std::cerr << "OK!\n";
    }
    catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STIR
