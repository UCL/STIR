//
//
/*!
  \file
  \ingroup examples
  \brief A modification of demo2.cxx that parses all parameters from a parameter file.

  It illustrates
        - basic class derivation principles
        - how to use ParsingObject to have automatic capabilities of parsing
          parameters files (and interactive questions to the user)
        - how most STIR programs parse the parameter files.

  Note that the same functionality could be provided without deriving
  a new class from ParsingObject. One could have a KeyParser object
  in main() and fill it in directly.

  See README.txt in the directory where this file is located.

  \author Kris Thielemans
*/
/*
    Copyright (C) 2004- 2012, Hammersmith Imanet Ltd

    This software is distributed under the terms
    of the GNU General  Public Licence (GPL)
    See STIR/LICENSE.txt for details
*/
#include <memory>
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"
#include "stir/Succeeded.h"
#include "stir/display.h"
#include <stdio.h>

namespace stir
{

class MyStuff : public ParsingObject
{
public:
  void set_defaults();
  void initialise_keymap();
  void run(const bool display_off);

private:
  std::string input_filename;
  std::string template_filename;
  shared_ptr<BackProjectorByBin> back_projector_sptr;
  shared_ptr<OutputFileFormat<DiscretisedDensity<3, float>>> output_file_format_sptr;
};

void
MyStuff::set_defaults()
{
  auto projection_matrix_sptr = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  back_projector_sptr = std::make_shared<BackProjectorByBinUsingProjMatrixByBin>(projection_matrix_sptr);
  output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3, float>>::default_sptr();
}

void
MyStuff::initialise_keymap()
{
  parser.add_start_key("MyStuff parameters");
  parser.add_key("input file", &input_filename);
  parser.add_key("template image file", &template_filename);
  parser.add_parsing_key("back projector type", &back_projector_sptr);
  parser.add_parsing_key("output file format type", &output_file_format_sptr);
  parser.add_stop_key("End");
}

void
MyStuff::run(const bool display_off)
{

  shared_ptr<ProjData> proj_data_sptr(ProjData::read_from_file(input_filename));
  shared_ptr<ProjDataInfo> proj_data_info_sptr(proj_data_sptr->get_proj_data_info_sptr()->clone());

  shared_ptr<DiscretisedDensity<3, float>> density_sptr(read_from_file<DiscretisedDensity<3, float>>(template_filename));

  density_sptr->fill(0);

  /////////////// back project
  back_projector_sptr->set_up(proj_data_info_sptr, density_sptr);

  back_projector_sptr->back_project(*density_sptr, *proj_data_sptr);

  /////////////// output
  output_file_format_sptr->write_to_file("demo3_density", *density_sptr);

  if (!display_off)
    display(*density_sptr, density_sptr->find_max(), "Output");
}

} // end of namespace stir

int
main(int argc, char** argv)
{
  using namespace stir;

  MyStuff my_stuff;
  my_stuff.set_defaults();
  bool display_off = false;
  if (argc < 2)
    {
      std::cerr << "Normal usage: " << argv[0] << " parameter-file [--display_off]\n";
      std::cerr << "I will now ask you the questions interactively\n";
      my_stuff.ask_parameters();
    }
  else
    {
      my_stuff.parse(argv[1]);
      if (argc >= 3)
        // Set the display_off to true if the second argument is "--display_off"
        display_off = (strcmp(argv[2], "--display_off") == 0);
    }
  my_stuff.run(display_off);
  return EXIT_SUCCESS;
}
