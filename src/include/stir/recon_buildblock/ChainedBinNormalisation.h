//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class ChainedBinNormalisation

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_ChainedBinNormalisation_H__
#define __stir_recon_buildblock_ChainedBinNormalisation_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup recon_buildblock
  \brief A BinNormalisation class that simply multiplies the factors given by
  2 BinNormalisation objects.

  This is especially useful to combine the 'usual' normalisation factors and attenuation factors 
  in PET. As both are multiplicative corrections, they both belong in the BinNormalisation
  hierarchy.

  \par Parsing details
  \verbatim
  Chained Bin Normalisation Parameters:=
  ; type of one of the bin normalisations, followed by its parameters
  Bin Normalisation to apply first := <ASCII>

  ; type of the other, and its parameters
  Bin Normalisation to apply second := <ASCII>
  END Chained Bin Normalisation Parameters :=
  \endverbatim
  \par Example
  This example shows how to construct the parameter file for the case that there
  are normalisation factors in a file \a norm.hs and an attenuation image in a file 
  \a atten.hv.
  \see BinNormalisationFromProjData, BinNormalisationFromAttenuationImage.

  \verbatim
  Bin Normalisation type := Chained
  Chained Bin Normalisation Parameters:=
    Bin Normalisation to apply first := from projdata
      Bin Normalisation From ProjData :=
        normalisation projdata filename:= norm.hs
      End Bin Normalisation From ProjData:= 
    Bin Normalisation to apply second := From Attenuation Image
      Bin Normalisation From Attenuation Image:=
        attenuation_image_filename := atten.hv
        forward projector type := ray tracing
          Forward Projector Using Ray Tracing Parameters :=
          End Forward Projector Using Ray Tracing Parameters :=
      End Bin Normalisation From Attenuation Image :=
  END Chained Bin Normalisation Parameters :=
  \endverbatim
*/
class ChainedBinNormalisation :
   public RegisteredParsingObject<ChainedBinNormalisation, BinNormalisation>
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
  ChainedBinNormalisation();

ChainedBinNormalisation(shared_ptr<ChainedBinNormalisation> const& apply_first,
		        shared_ptr<ChainedBinNormalisation> const& apply_second);

  //! Checks if we can handle certain projection data.
  /*! Calls set_up for the BinNormalisation members. */
  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);

  //! Normalise some data
  /*! 
    This calls apply() of the 2 BinNormalisation members
  */
  virtual void apply(RelatedViewgrams<float>& viewgrams) const;

  //! Undo the normalisation of some data
  /*! 
    This calls undo() of the 2 BinNormalisation members. 
  */
  virtual void undo(RelatedViewgrams<float>& viewgrams) const;

  virtual float get_bin_efficiency(const Bin& bin) const;
 

private:
  shared_ptr<ChainedBinNormalisation> apply_first;
  shared_ptr<ChainedBinNormalisation> apply_second;
  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
