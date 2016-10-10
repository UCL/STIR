//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationFromAttenuationImage

  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_BinNormalisationFromAttenuationImage_H__
#define __stir_recon_buildblock_BinNormalisationFromAttenuationImage_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
  \brief A BinNormalisation class that gets attenuation correction factors from
  an attenuation image

  This forwards projects the attenuation image, multiplies with -1, and exponentiates
  to obtain the attenuation correction factors.  

  Default forward projector is ForwardProjectorByBinUsingRayTracing.

  \warning Attenuation image data are supposed to be in units cm^-1. 
    (Reference: water has mu .096 cm^-1.)
  \todo Add mechanism for caching the attenuation correction factors, such that they will
  be calculated only once. However, caching should by default be disabled, as most 
  applications need them only once anyway.

  \par Parsing details
  \verbatim
  Bin Normalisation From Attenuation Image:=
  attenuation_image_filename := <ASCII>
  forward projector type := <ASCII>
  End Bin Normalisation From Attenuation Image :=
  \endverbatim
*/
class BinNormalisationFromAttenuationImage :
   public RegisteredParsingObject<BinNormalisationFromAttenuationImage, BinNormalisation>
{
public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char * const registered_name; 
  
  //! Default constructor
  /*! 
    \warning You should not call any member functions for any object just 
    constructed with this constructor. Initialise the object properly first
    by parsing.
  */
  BinNormalisationFromAttenuationImage();

  //! Constructor that reads the image from a file
  /*! Default forward projector is ForwardProjectorByBinUsingRayTracing. */
  BinNormalisationFromAttenuationImage(const std::string& filename, shared_ptr<ForwardProjectorByBin> const& =shared_ptr<ForwardProjectorByBin>());

  //! Constructor that takes the image as an argument
  /*! Default forward projector is ForwardProjectorByBinUsingRayTracing.
      The image pointed to by attenuation_image_ptr is NOT modified.
  */
  BinNormalisationFromAttenuationImage(shared_ptr<DiscretisedDensity<3,float> > const& attenuation_image_ptr,
                                       shared_ptr<ForwardProjectorByBin> const& = shared_ptr<ForwardProjectorByBin>());

  //! Checks if we can handle certain projection data.
  /*! This test is essentially checking if the forward projector can handle the data 
      by calling ForwardProjectorByBin::set_up().
  */
  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);

  //! Normalise some data
  /*! 
    This means \c multiply with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  //! Undo the normalisation of some data
  /*! 
    This means \c divide with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  virtual float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const; 

private:
  shared_ptr<DiscretisedDensity<3,float> > attenuation_image_ptr;
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr;

  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  std::string attenuation_image_filename;

protected:
  virtual
  std::vector<float> get_related_bins_values(const std::vector<Bin>&) const;
};


END_NAMESPACE_STIR

#endif
