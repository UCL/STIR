//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of class BinNormalisationFromProjData

  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisationFromProjData_H__
#define __stir_recon_buildblock_BinNormalisationFromProjData_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR

class ProjData;
template <typename T> class shared_ptr;

/*!
  \ingroup recon_buildblock
  \brief A BinNormalisation class that gets the normalisation factors from
  a ProjData object

  \warning the ProjData object containing the normalisation factors should 
  currently have exactly the same dimensions as the data it is applied on.
  
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
  BinNormalisationFromProjData(const string& filename);

  //! Constructor that takes the projdata from a shared_pointer
  /*! The projdata object pointed to will \c not be modified. */
  BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr);

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
  shared_ptr<ProjData> norm_proj_data_ptr;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  string normalisation_projdata_filename;
};


END_NAMESPACE_STIR

#endif
