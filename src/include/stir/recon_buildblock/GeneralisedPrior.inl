//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief Inline implementations for class GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/

START_NAMESPACE_TOMO


template <typename elemT>
GeneralisedPrior<elemT>::GeneralisedPrior() 
{ 
  penalisation_factor =0;
}


template <typename elemT>
float
GeneralisedPrior<elemT>::
get_penalisation_factor() const
{ return penalisation_factor; }

template <typename elemT>
void
GeneralisedPrior<elemT>::
set_penalisation_factor(const float new_penalisation_factor)
{ penalisation_factor = new_penalisation_factor; }

END_NAMESPACE_TOMO

