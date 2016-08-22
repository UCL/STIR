/*
  Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \ingroup buildblock
  \ingroup scatter
  \brief Compute a mask for the "tails" in the sinogram

  This computes the outer part of the projection data that
  does not receive any (unscattered) contributions from inside the body.
  Its normal input is a projection data with attenuation-correction factors
  (ACFs). We take a threshold as parameter (which should be a bit larger
  than 1). Along each line in the sinogram, we search from the edges
  until we find the first bin above the threshold. We then go back a
  few bins (as set by \a safety-margin). All bins from the edge until
  this point are included in the mask.

  As from 23 July 2016, the functionality of this executable was transfered in
  a new class CreateTailMaskFromACFs. This made possible to use par file to
  initiliase the process and use it from within some other code.

  In the later case, were the output is to be used from some other code the
  output file can be omitted, in order to avoid unessery outputs.

  \author Nikos Efthimiou
  \author Kris Thielemans

  \par Usage:

  \verbatim
   create_tail_mask_from_ACFs --ACF-filename <filename> \\
        --output-filename <filename> \\
        [--ACF-threshold <float>] \\
        [--safety-margin <integer>]
  \endverbatim

  \par Alternative Usage:

  \verbatim
    create_tail_mask_from_ACFs <filename.par>
  \endverbatim

  \par Example of parameter file:
  \verbatim
    CreateTailMaskFromACFs :=
        ACF-filename :=
        output-filename :=
        ACF-threshold :=
        safety-margin :=
    END CreateTailMaskFromACFs :=
  \endverbatim
  ACF-threshold defaults to 1.1 (should be larger than 1), safety-margin to 4
*/

#ifndef __stir_scatter_CreateTailMaskFromACFs_H__
#define __stir_scatter_CreateTailMaskFromACFs_H__

#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"


#include "stir/ParsingObject.h"
#include "stir/ProjDataInMemory.h"

START_NAMESPACE_STIR

//!
//! \brief The CreateTailMaskFromACFs class
//! \author Nikos Efthimiou
//! \details This class implements the functionality of the executable.
//! It was nessesary in order to be able to perform this calculations from
//! the ScatterEstimationByBin.
class CreateTailMaskFromACFs : public ParsingObject
{
public:
    CreateTailMaskFromACFs();

    virtual Succeeded process_data();

    void set_input_projdata(shared_ptr<ProjData> &);

    void set_input_projdata(std::string&);

    void set_output_projdata(shared_ptr<ProjData>&);

    void set_output_projdata(std::string&);

    //!
    //! \brief get_output_projdata
    //! \return
    //! \details Use this function to return the output
    //! projdata.
    shared_ptr<ProjData> get_output_projdata();

    //!
    //! \brief ACF_threshold
    //! \warning ACF-threshold defaults to 1.1 (should be larger than 1)
    float ACF_threshold;

    //!
    //! \brief safety_margin
    //!
    int safety_margin;

protected:
    void initialise_keymap();
    bool post_processing();
    void set_defaults();

private:
    //!
    //! \brief ACF_sptr
    //! \details Input projdata
    shared_ptr<ProjData> ACF_sptr;

    //!
    //! \brief mask_proj_data
    //! \details Output projdata
    shared_ptr<ProjData> mask_proj_data;

    //!
    //! \brief _input_filename
    //! \details The input filename can be omitted in the par file
    //! but has to be set, later, using the set_input_projdata().
    std::string _input_filename;

    //!
    //! \brief _output_filename
    //! \details This is the output filename.
    //! It can be omited, if an output is not nessesary.
    std::string _output_filename;
};

END_NAMESPACE_STIR

#endif
