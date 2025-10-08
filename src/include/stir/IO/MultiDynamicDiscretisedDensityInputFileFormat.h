//
//
#ifndef __stir_IO_MultiDynamicDiscretisedDensityInputFileFormat_h__
#define __stir_IO_MultiDynamicDiscretisedDensityInputFileFormat_h__
/*
    Copyight (C) 2018,2020, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::MultiDynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

//! Class for reading images in Multi file-format.
/*! \ingroup IO

*/
class MultiDynamicDiscretisedDensityInputFileFormat : public InputFileFormat<DynamicDiscretisedDensity>
{
public:
  const std::string get_name() const override { return "Multi"; }

protected:
  bool actual_can_read(const FileSignature& signature, std::istream&) const override;
  //! always throws via error()
  std::unique_ptr<data_type> read_from_file(std::istream&) const override;
  std::unique_ptr<data_type> read_from_file(const std::string& filename) const override;
};
END_NAMESPACE_STIR

#endif
