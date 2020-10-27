//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationFromProjData

  \author Kris Thielemans
*/
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_BinNormalisationFromProjData_H__
#define __stir_recon_buildblock_BinNormalisationFromProjData_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
  \brief A BinNormalisation class that gets the normalisation factors from
  a ProjData object

  \warning the ProjData object containing the normalisation factors should 
  currently have exactly the same dimensions as the data it is applied on.

  \par Parsing details
  \verbatim
  Bin Normalisation From ProjData:=
  normalisation projdata filename := <ASCII>
  End Bin Normalisation From ProjData:=
  \endverbatim
*/
class BinNormalisationFromProjData :
   public RegisteredParsingObject<BinNormalisationFromProjData, BinNormalisation>
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
  BinNormalisationFromProjData();

  //! Constructor that reads the projdata from a file
  BinNormalisationFromProjData(const std::string& filename);

  //! Constructor that takes the projdata from a shared_pointer
  /*! The projdata object pointed to will \c not be modified. */
  BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr);

  //! check if we would be multiplying with 1 (i.e. do nothing)
  /*! Checks if all data is equal to 1 (up to a tolerance of 1e-4). To do this, it uses ProjData::get_viewgram()
      and loops over all viewgrams.
  */
  virtual bool is_trivial() const;

  //! Checks if we can handle certain projection data.
  /*! Compares the  ProjDataInfo from the ProjData object containing the normalisation factors 
      with the ProjDataInfo supplied. */
  virtual Succeeded set_up(const shared_ptr<const ProjDataInfo>&);

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
    //! Get a shared_ptr to the normalisation proj_data.
  virtual shared_ptr<ProjData> get_norm_proj_data_sptr() const;
 
private:
  shared_ptr<ProjData> norm_proj_data_ptr;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  std::string normalisation_projdata_filename;
};


END_NAMESPACE_STIR

#endif
