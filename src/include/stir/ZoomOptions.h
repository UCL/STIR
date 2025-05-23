/*
    Copyright (C) 2018-2019, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_ZOOMOPTIONS_H__
#define  __stir_ZOOMOPTIONS_H__

/*!
  \file

  \brief Declaration of class stir::ZoomOptions
  \ingroup buildblock

  \author Ludovica Brusaferri
  \author Kris Thielemans
  
*/

#include "stir/error.h"

START_NAMESPACE_STIR

/*!
  \brief
  This class enables the user to choose between different zooming options
  \ingroup buildblock
  
  The 3 possible values determine a global scale factor used for the end result:
  (i) preserve sum (locally)
  (ii) preserve values (like interpolation)
  (iii) preserve projections: using a STIR forward projector on the zoomed image will give (approximately) the same projections.
  
  \see zoom_image
*/

class ZoomOptions{
 public:
  enum Scaling {preserve_sum, preserve_values, preserve_projections};
  //! constructor from Scaling
  /*! calls error() if out-of-range
   */
  ZoomOptions(const Scaling v = preserve_sum) : v(v)
    {
      // need to catch out-of-range in case somebody did a static_cast from an int (e.g. SWIG does)
      if ((v < preserve_sum) || (v > preserve_projections))
        error("ZoomOptions initialised with out-of-range value");
    }
  Scaling get_scaling_option() const { return v; }
 private:
  Scaling v;
};

END_NAMESPACE_STIR


#endif // ZOOMOPTIONS_H
