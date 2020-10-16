//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class ChainedBinNormalisation

  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2005, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_ChainedBinNormalisation_H__
#define __stir_recon_buildblock_ChainedBinNormalisation_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
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

ChainedBinNormalisation(shared_ptr<BinNormalisation> const& apply_first,
		        shared_ptr<BinNormalisation> const& apply_second);

  //! Checks if we can handle certain projection data.
  /*! Calls set_up for the BinNormalisation members. */
  virtual Succeeded set_up(const shared_ptr<const ProjDataInfo>&);

  //! Normalise some data
  /*! 
    This calls apply() of the 2 BinNormalisation members
  */
  virtual void apply(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  virtual void apply(ProjData&,const double start_time, const double end_time) const;

virtual void apply_only_first(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

virtual void apply_only_first(ProjData&,const double start_time, const double end_time) const;

virtual void apply_only_second(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

virtual void apply_only_second(ProjData&,const double start_time, const double end_time) const;

  //! Undo the normalisation of some data
  /*! 
    This calls undo() of the 2 BinNormalisation members. 
  */
  virtual void undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

  virtual void undo(ProjData&,const double start_time, const double end_time) const;

virtual void undo_only_first(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

virtual void undo_only_first(ProjData&,const double start_time, const double end_time) const;

virtual void undo_only_second(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const;

virtual void undo_only_second(ProjData&,const double start_time, const double end_time) const;

  virtual float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const;
 //! Returns the is_trivial() status of the first normalisation object.
 //! \warning Currently, if the object has not been set the function throws an error.
  virtual bool is_first_trivial() const;
//! Returns the is_trivial() status of the second normalisation object.
//! \warning Currently, if the object has not been set the function throws an error.
  virtual bool is_second_trivial() const;

  virtual shared_ptr<BinNormalisation> get_first_norm() const;

  virtual shared_ptr<BinNormalisation> get_second_norm() const;
private:
  shared_ptr<BinNormalisation> apply_first;
  shared_ptr<BinNormalisation> apply_second;
  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
