//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
/*!

  \file
  \ingroup symmetries

  \brief Declaration of class stir::DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project


*/
#ifndef __stir_recon_buildblock_DataSymmetriesForBins_H__
#define __stir_recon_buildblock_DataSymmetriesForBins_H__

#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ProjDataInfo.h" // is used in .inl
#include "stir/shared_ptr.h"
#include <vector>
#include <memory>

#include "stir/Coordinate2D.h"

#ifndef STIR_NO_NAMESPACES
using std::vector;
#ifndef STIR_NO_AUTO_PTR
using std::auto_ptr;
#endif
#endif

START_NAMESPACE_STIR

class Bin;
class SymmetryOperation;
class ProjDataInfo;


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
  \ingroup symmetries
  \brief A class for encoding/finding symmetries common to the geometry
  of the projection data and the discretised density. 

  This class is mainly (only?) useful for ProjMatrixByBin classes and their
  'users'. Together with SymmetryOperation, it provides the basic 
  way to be able to write generic code without knowing which 
  particular symmetries the data have.

  \todo I have used Bin here to have the 4 coordinates, but Bin has data as well 
  which is not really necessary here.
*/
class DataSymmetriesForBins : public DataSymmetriesForViewSegmentNumbers
{
private:
  typedef DataSymmetriesForViewSegmentNumbers base_type;
  typedef DataSymmetriesForBins self_type;

public:
  DataSymmetriesForBins(const shared_ptr<ProjDataInfo>& proj_data_info_ptr);

  virtual ~DataSymmetriesForBins();

  virtual 
#ifndef STIR_NO_COVARIANT_RETURN_TYPES
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

  /*! \brief given an arbitrary bin 'b', find the basic bin and the corresponding symmetry operation
  
  sets 'b' to the corresponding 'basic' bin and returns the symmetry 
  transformation from 'basic' to 'b'.
  */
  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_bin(Bin&) const = 0;

  /*! \brief given an arbitrary bin 'b', find the basic bin
  
  sets 'b' to the corresponding 'basic' bin and returns true if
  'b' is changed (i.e. it was NOT a basic bin).
  */
  virtual bool
    find_basic_bin(Bin& b) const;

  /*! \brief test if a bin is 'basic' 

  The default implementation uses find_basic_bin
  */
  virtual bool
    is_basic(const Bin& v_s) const;

  //! default implementation in terms of find_symmetry_operation_from_basic_bin
  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_view_segment_numbers(ViewSegmentNumbers&) const;


protected:
  //! Member storing the info needed by get_related_bins() et al
  const shared_ptr<ProjDataInfo> proj_data_info_ptr;

  //! Check equality
  virtual bool blindly_equals(const root_type * const) const = 0;
};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/DataSymmetriesForBins.inl"


#endif

