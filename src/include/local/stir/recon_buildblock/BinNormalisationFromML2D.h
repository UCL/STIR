//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisationFromML2D

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisationFromML2D_H__
#define __stir_recon_buildblock_BinNormalisationFromML2D_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInMemory.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup recon_buildblock
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
class BinNormalisationFromML2D :
   public RegisteredParsingObject<BinNormalisationFromML2D, BinNormalisation>
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
  BinNormalisationFromML2D();

  //! Checks if we can handle certain projection data.
  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);

  //! Normalise some data
  /*! 
    This means \c multiply with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void apply(RelatedViewgrams<float>& viewgrams) const;

  //! Undo the normalisation of some data
  /*! 
    This means \c divide with the data in the projdata object 
    passed in the constructor. 
  */
  virtual void undo(RelatedViewgrams<float>& viewgrams) const;

private:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  string normalisation_filename_prefix;
  bool do_block;
  bool do_geo;
  bool do_eff;
  int eff_iter_num;
  int iter_num;
  // use shared pointer to avoid calling default constructor
  shared_ptr<ProjDataInMemory> norm_factors_ptr;
};


END_NAMESPACE_STIR

#endif
