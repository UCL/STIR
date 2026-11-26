/*
    Copyright (C) 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup main_programs
  \brief main() for stir::OSMAPOSLReconstruction on parametric images

  \author Nicolas A Karakatsanis
*/
#include "stir/Succeeded.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/recon_buildblock/distributable_main.h"

#include "stir/is_null_ptr.h"
#include <fstream>
USING_NAMESPACE_STIR

//! OSMAPOSL for generalized Patlak, allowing a 2-parameter initial estimate
/*! \ingroup recon_buildblock
  The generalized Patlak reconstruction uses a 3-parameter target
  (\c [slope, kloss, intercept]), so \c IterativeReconstruction::get_initial_data_ptr()
  can only read a 3-parameter initial estimate

  In the reconstruction tests (and in practice, where a standard Patlak result is the
  starting point) we want to initialise from a 2-parameter image. This class
  overrides \c get_initial_data_ptr() to peek at the number of parameters in the file.

*/
class NestedGeneralizedPatlakOSMAPOSL : public OSMAPOSLReconstruction<Parametric3VoxelsOnCartesianGrid>
{
private:
  typedef OSMAPOSLReconstruction<Parametric3VoxelsOnCartesianGrid> base_type;

  int find_num_image_data_types(std::ifstream& input) const
  {
    const std::string key = "number of image data types";
    std::string line;
    while (std::getline(input, line))
      {
        const auto key_pos = line.find(key);
        if (key_pos == std::string::npos)
          continue;
        const auto eq_pos = line.find(":=", key_pos);
        if (eq_pos == std::string::npos)
          continue;
        return std::atoi(line.c_str() + eq_pos + 2);
      }
    return -1;
  }

public:
  using base_type::base_type; // inherit constructors, so the argv[1] form still works

  Parametric3VoxelsOnCartesianGrid* get_initial_data_ptr() const override
  {

    std::ifstream image_stream(this->initial_data_filename.c_str());
    const int file_num_params = this->initial_data_filename.empty() ? -1 : find_num_image_data_types(image_stream);

    if (file_num_params != 2)
      return base_type::get_initial_data_ptr(); // uniform, or a normal 3-parameter file

    info("Initialising generalized Patlak from a 2-parameter Patlak image; kloss initialised to zero.");

    const auto par2 = Parametric2VoxelsOnCartesianGrid::read_from_file(this->initial_data_filename);

    if (is_null_ptr(this->objective_function_sptr))
      error("objective function needs to be set before calling get_initial_data_ptr");

    const auto slope = par2->construct_single_density(1);
    const auto intercept = par2->construct_single_density(2);
    auto kloss = slope;
    kloss.fill(0.F);

    // this constructor takes geometry, exam info and timing from the single image
    auto par3 = new Parametric3VoxelsOnCartesianGrid(slope);
    par3->update_parametric_image(slope, 1);
    par3->update_parametric_image(kloss, 2);
    par3->update_parametric_image(intercept, 3);
    return par3;
  }
};

#ifdef STIR_MPI
int
stir::distributable_main(int argc, char** argv)
#else
int
main(int argc, char** argv)
#endif
{

  HighResWallClockTimer t;
  t.reset();
  t.start();

  NestedGeneralizedPatlakOSMAPOSL reconstruction_object(argc > 1 ? argv[1] : "");

  if (reconstruction_object.reconstruct() == Succeeded::yes)
    {
      t.stop();
      std::cout << "Total Wall clock time: " << t.value() << " seconds" << std::endl;
      return EXIT_SUCCESS;
    }
  else
    {
      t.stop();
      return EXIT_FAILURE;
    }
}
