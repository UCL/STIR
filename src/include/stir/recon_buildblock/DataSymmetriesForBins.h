//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Declaration of class DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#ifndef __DataSymmetriesForBins_H__
#define __DataSymmetriesForBins_H__

#include "DataSymmetriesForViewSegmentNumbers.h"
#include "ProjDataInfo.h"
#include "shared_ptr.h"
#include <vector>
#include <memory>

#include "Coordinate2D.h"

#ifndef TOMO_NO_NAMESPACES
using std::vector;
#ifndef TOMO_NO_AUTO_PTR
using std::auto_ptr;
#endif
#endif

START_NAMESPACE_TOMO

class Bin;
class SymmetryOperation;


#if 0
class BinIndexRange;
#endif

/*! 
  \brief AxTangPosNumbers as a class that provides the 2 remaining
  coordinates for a Bin, aside from ViewSegmentNumbers.

  TODO? something more sofisticated than a typedef
*/
typedef Coordinate2D<int> AxTangPosNumbers;


/*!
  \ingroup recon_buildblock
  \brief A class for encoding/finding symmetries common to the geometry
  of the projection data and the discretised density. 

  This class is mainly (only?) useful for ProjMatrixByBin classes and their
  'users'. Together with SymmetryOperation, it provides the basic 
  way to be able to write generic code without knowing which 
  particular symmetries the data have.

  TODO? I've used Bin here to have the 4 coordinates, but Bin has data as well 
  which is not really necessary here.
*/
class DataSymmetriesForBins : public DataSymmetriesForViewSegmentNumbers
{
public:
  DataSymmetriesForBins(const shared_ptr<ProjDataInfo>& proj_data_info_ptr);

  virtual ~DataSymmetriesForBins() {};

  virtual 
#ifndef TOMO_NO_COVARIANT_RETURN_TYPES
    DataSymmetriesForBins 
#else
    DataSymmetriesForViewSegmentNumbers
#endif
    * clone() const = 0;

#if 0
  TODO!
  //! returns the range of the indices for basic bins
  virtual BinIndexRange
    get_basic_bin_index_range() const = 0;
#endif

  //! fills in a vector with all the bins that are related to 'b' (including itself)
  /*! range for axial_pos_num and tangential_pos_num is taken from the ProjDataInfo object 
      passed in the constructor 
      \warning \c b has to be a 'basic' bin
  */
  // next return value could be a RelatedBins ???
  // however, both Bin and RelatedBins have data in there (which is not needed here)
  inline void
    get_related_bins(vector<Bin>&, const Bin& b) const;

  //! fills in a vector with all the bins (within the range) that are related to 'b'
  /*! \warning \c b has to be a 'basic' bin
  */
  virtual void
    get_related_bins(vector<Bin>&, const Bin& b,
                      const int min_axial_pos_num, const int max_axial_pos_num,
                      const int min_tangential_pos_num, const int max_tangential_pos_num) const;

  //! fills in a vector with the axial and tangential position numbers related to this bin
  /*! range for axial_pos_num and tangential_pos_num is taken from the ProjDataInfo object 
      passed in the constructor 
      \warning \c b has to be a 'basic' bin
      \see 6 argument version of get_related_bins_factorised()
  */
  inline void
    get_related_bins_factorised(vector<AxTangPosNumbers>&, const Bin& b) const;

 //! fills in a vector with the axial and tangential position numbers related to this bin
  /*!
     It is guaranteed (or at least, it should be by the implementation of the derived class)
     that these AxTangPosNumbers are related for all related ViewSegmentNumbers
     for this bin.

     So, you can find all related bins by calling get_related_ViewSegmentNumbers()
     and get_related_bins_factorised(), which is what the default implementation 
     does. (A derived class might do this in a more optimal way.)

     \warning \c b has to be a 'basic' bin
  */
  virtual void
    get_related_bins_factorised(vector<AxTangPosNumbers>&, const Bin& b,
                                const int min_axial_pos_num, const int max_axial_pos_num,
                                const int min_tangential_pos_num, const int max_tangential_pos_num) const = 0;

  //! returns the number of bins related to 'b'
  virtual int
    num_related_bins(const Bin& b) const;

  /*! \brief given an arbitrary bin 'b', find the basic bin
  
  sets 'b' to the corresponding 'basic' bin and returns the symmetry 
  transformation from 'basic' to 'b'.
  */
  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_to_basic_bin(Bin&) const = 0;

  /*! \brief given an arbitrary bin 'b', find the basic bin
  
  sets 'b' to the corresponding 'basic' bin and returns true if
  'b' is changed (i.e. it was NOT a basic bin).
  */
  virtual bool
    find_basic_bin(Bin& b) const;

  //! default implementation in terms of find_symmetry_operation_to_basic_bin
  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_to_basic_view_segment_numbers(ViewSegmentNumbers&) const;


protected:
  //! Member storing the info needed by get_related_bins() et al
  const shared_ptr<ProjDataInfo> proj_data_info_ptr;
};

END_NAMESPACE_TOMO

#include "recon_buildblock/DataSymmetriesForBins.inl"


#endif

