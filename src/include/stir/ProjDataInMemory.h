//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ProjDataInMemory

  \author Kris Thielemans

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __ProjDataInMemory_H__
#define __ProjDataInMemory_H__

#include "stir/ProjDataFromStream.h" 
#include <string>
#ifdef BOOST_NO_STRINGSTREAM
#include <memory>

#ifndef STIR_NO_NAMESPACES
#ifndef TOMO_NO_AUTOPTR
using std::auto_ptr;
#endif
#endif
#endif // BOOST_NO_STRINGSTREAM

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup buildblock
  \brief A class which reads/writes projection data from/to memory.

  Mainly useful for temporary storage of projection data.

*/
class ProjDataInMemory : public ProjDataFromStream
{
public: 
    
  //! constructor with only info, but no data
  /*! 
    \param proj_data_info_ptr object specifying all sizes etc.
      The ProjDataInfo object pointed to will not be modified.
    \param initialise_with_0 specifies if the data should be set to 0. 
        If \c false, the data is undefined until you set it yourself.
  */
  ProjDataInMemory (shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
                    const bool initialise_with_0 = true);

  //! constructor that copies data from another ProjData
  ProjDataInMemory (const ProjData& proj_data);

  ~ProjDataInMemory();

  //! writes info to a file in Interfile format
  /*! \warning This will become obsolete as soon as we have proper output of projdata
  */
  Succeeded
    write_to_file(const string& filename) const;
    
private:
#ifdef BOOST_NO_STRINGSTREAM
  auto_ptr<char> buffer;
#else
#endif
  
  size_t get_size_of_buffer() const;
};

END_NAMESPACE_STIR


#endif
