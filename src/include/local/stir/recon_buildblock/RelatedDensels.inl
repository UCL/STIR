//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementations of inline functions of class RelatedDensels

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$
*/
#include "local/tomo/Densel.h"
#include "local/tomo/recon_buildblock/DataSymmetriesForDensels.h"

START_NAMESPACE_TOMO



RelatedDensels::RelatedDensels()
:related_densels(),symmetries()
{}

RelatedDensels::RelatedDensels(const vector< Densel>& related_densels_v,
                         const shared_ptr<DataSymmetriesForDensels>& symmetries_used)
:related_densels(related_densels_v),symmetries(symmetries_used)
{}

int
RelatedDensels::get_num_related_densels() const
{
  return related_densels.size();
}

Densel
RelatedDensels::get_basic_densel() const
{
  assert(related_densels.size() != 0);
  return related_densels[0];
}


#if 0
const ProjDataInfo *
RelatedDensels:: get_proj_data_info_ptr() const
{
 
  return related_densels[0].get_proj_data_info_ptr();
}
#endif


const DataSymmetriesForDensels*
RelatedDensels::get_symmetries_ptr() const
{
  return symmetries.get();
}


RelatedDensels::iterator 
RelatedDensels::begin()
{ return related_densels.begin();}

RelatedDensels::iterator
RelatedDensels::end()
{return related_densels.end();}

RelatedDensels::const_iterator 
RelatedDensels::begin() const
{return related_densels.begin();}

RelatedDensels::const_iterator 
RelatedDensels::end() const
{return related_densels.end();}


END_NAMESPACE_TOMO

