//
// $Id$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Declaration of class DataSymmetriesForBins_PET_CartesianGrid

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#ifndef __DataSymmetriesForBins_PET_CartesianGrid_H__
#define __DataSymmetriesForBins_PET_CartesianGrid_H__


#include "recon_buildblock/DataSymmetriesForBins.h"
//#include "SymmetryOperations_PET_CartesianGrid.h"
//#include "ViewSegmentNumbers.h"
//#include "VoxelsOnCartesianGrid.h"
#include "Bin.h"

START_NAMESPACE_TOMO

template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <int num_dimensions, typename elemT> class DiscretisedDensityOnCartesianGrid;
template <typename T> class shared_ptr;

/*!
  \ingroup recon_buildblock
  \brief Symmetries appropriate for a (cylindrical) PET scanner, and 
  a discretised density on a Cartesian grid.

  All operations (except the constructor) are inline as timing of
  the methods of this class is critical.
*/
class DataSymmetriesForBins_PET_CartesianGrid : public DataSymmetriesForBins
{
public:

  DataSymmetriesForBins_PET_CartesianGrid(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
                                            const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr);


  virtual 
    inline 
#ifndef TOMO_NO_COVARIANT_RETURN_TYPES
    DataSymmetriesForBins 
#else
    DataSymmetriesForViewSegmentNumbers
#endif
    * clone() const;
#if 0
  TODO!
  //! returns the range of the indices for basic bins
  virtual BinIndexRange
    get_basic_bin_index_range() const;
#endif

  inline void
    get_related_bins_factorised(vector<AxTangPosNumbers>&, const Bin& b,
                                const int min_axial_pos_num, const int max_axial_pos_num,
                                const int min_tangential_pos_num, const int max_tangential_pos_num) const;

  inline int
    num_related_bins(const Bin& b) const;

  inline auto_ptr<SymmetryOperation>
    find_symmetry_operation_to_basic_bin(Bin&) const;

  inline bool
    find_basic_bin(Bin& b) const;
  
  inline int
    num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const;
  
  inline void
    get_related_view_segment_numbers(vector<ViewSegmentNumbers>& rel_vs, const ViewSegmentNumbers& vs) const;
  
  inline bool
    find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const;

  //! find out how many image planes there are for every scanner ring
  inline float get_num_planes_per_scanner_ring() const;



  //! find correspondence between axial_pos_num and image coordinates
  /*! z = num_planes_per_axial_pos * axial_pos_num + axial_pos_to_z_offset
  
      compute the offset by matching up the centre of the scanner 
      in the 2 coordinate systems
      */
  inline float get_num_planes_per_axial_pos(const int segment_num) const;
  inline float get_axial_pos_to_z_offset(const int segment_num) const;
  
private:
  //const shared_ptr<ProjDataInfo>& proj_data_info_ptr;
  int num_views;
  int num_planes_per_scanner_ring;
  //! a list of values for every segment_num
  VectorWithOffset<int> num_planes_per_axial_pos;
  //! a list of values for every segment_num
  VectorWithOffset<float> axial_pos_to_z_offset;

#if 0
  // at the moment, we don't need the following 2 members

  // TODO somehow store only the info
  shared_ptr<DiscretisedDensity<3,float> > image_info_ptr;

  // a convenience function that does the dynamic_cast from the above
  inline const DiscretisedDensityOnCartesianGrid<3,float> *
    cartesian_grid_info_ptr() const;
#endif

  inline bool
  find_basic_bin(int &segment_num, int &view_num, int &axial_pos_num, int &tangential_pos_num) const;

  
  inline int find_transform_z(
			 const int segment_num, 
			 const int  axial_pos_num) const;
  
  inline SymmetryOperation* 
    find_sym_op_general_bin(   
    int s, 
    int seg, 
    int view_num, 
    int axial_pos_num) const;
  
  inline SymmetryOperation* 
    find_sym_op_bin0(   
    int seg, 
    int view_num, 
    int axial_pos_num) const;
  
};

END_NAMESPACE_TOMO

#include "recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.inl"

#endif
