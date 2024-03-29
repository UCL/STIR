//
//

/*!
  \file
  \ingroup utilities

  \brief A utility to rebin projection data.


  Here's a sample .par file
\verbatim
rebin_projdata Parameters :=
  rebinning type := FORE
    FORE Parameters :=
    ...
    End FORE Parameters:=
END:=
\endverbatim

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ProjDataRebinning.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/is_null_ptr.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <iostream>

using std::cerr;
using std::endl;

START_NAMESPACE_STIR

// TODO most of this is identical to others, so make a common class
class RebinProjDataParameters : public ParsingObject
{
public:
  RebinProjDataParameters(const char* const par_filename);
  shared_ptr<ProjDataRebinning> proj_data_rebinning_sptr;

private:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

void
RebinProjDataParameters::set_defaults()
{
  proj_data_rebinning_sptr.reset();
}

void
RebinProjDataParameters::initialise_keymap()
{
  parser.add_start_key("Rebin_projdata Parameters");
  parser.add_parsing_key("rebinning type", &proj_data_rebinning_sptr);
  parser.add_stop_key("END");
}

bool
RebinProjDataParameters::post_processing()
{
  if (is_null_ptr(proj_data_rebinning_sptr))
    {
      warning("Invalid rebinning object\n");
      return true;
    }

  return false;
}

RebinProjDataParameters::RebinProjDataParameters(const char* const par_filename)
{
  set_defaults();
  Succeeded success = Succeeded::yes;
  if (par_filename != 0)
    success = parse(par_filename) == true ? Succeeded::yes : Succeeded::no;
  else
    ask_parameters();

  if (success == Succeeded::no || proj_data_rebinning_sptr->set_up() != Succeeded::yes)
    error("Rebin_projdata: set-up failed\n");
}

END_NAMESPACE_STIR

int
main(int argc, char* argv[])
{
  USING_NAMESPACE_STIR

  if (argc != 2)
    {
      cerr << "Usage: " << argv[0] << " par_file\n" << endl;
    }
  RebinProjDataParameters parameters(argc == 2 ? argv[1] : 0);

  if (argc != 2)
    {
      cerr << "Corresponding .par file input \n" << parameters.parameter_info() << endl;
    }

  return parameters.proj_data_rebinning_sptr->rebin() == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
