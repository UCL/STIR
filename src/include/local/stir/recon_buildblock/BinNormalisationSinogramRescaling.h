//
// $Id: 
//
/*!
  \file
  \ingroup local recon_buildblock

  \brief

  \author Sanida Mustafovic
  $Date: 
  $Revision: 
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#ifndef __stir_recon_buildblock_BinNormalisationSinogramRescaling_H__
#define __stir_recon_buildblock_BinNormalisationSinogramRescaling_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "local/stir/SinglesRates.h"
#include "stir/Scanner.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Array.h"
#include <string>

#ifndef STIR_NO_NAMESPACE
using std::string;
#endif

START_NAMESPACE_STIR


/*!
  \ingroup recon_buildblock
  \ingroup ECAT
  \brief A BinNormalisationSinogramRescaling class gets normaliastion factors by dividing 
   forward projection of the fitted cyl. to the precorrecred data
 
*/
class BinNormalisationSinogramRescaling :
   public RegisteredParsingObject<BinNormalisationSinogramRescaling, BinNormalisation>
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
  BinNormalisationSinogramRescaling();

  //! Constructor that reads the projdata from a file
  BinNormalisationSinogramRescaling(const string& filename);

  virtual Succeeded set_up(const shared_ptr<ProjDataInfo>&);
  float get_bin_efficiency(const Bin& bin, const double start_time, const double end_time) const;
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
  virtual void undo(RelatedViewgrams<float>& viewgrams, const double start_time, const double end_time) const;

private:
  // the proj data used for obtaining axial position num, segment num
  shared_ptr<ProjData> template_proj_data_sptr;
  Array<2,float> rescaling_factors;

  void read_rescaling_factors (const string& filename);
  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  string sinogram_rescaling_factors_filename;
  string template_proj_data_filename;
  ifstream instream;
};


END_NAMESPACE_STIR

#endif
