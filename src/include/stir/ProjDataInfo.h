//
// $Id$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class ProjDataInfo

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#ifndef __Tomo_ProjDataInfo_H__
#define __Tomo_ProjDataInfo_H__

#include "VectorWithOffset.h"
#include "Scanner.h"
#include "shared_ptr.h"
#include <string>

#ifndef TOMO_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_TOMO

template <typename elemT> class Sinogram;
template <typename elemT> class Viewgram;
template <typename elemT> class SegmentByView;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class RelatedViewgrams;
class DataSymmetriesForViewSegmentNumbers;
class ViewSegmentNumbers;
class Bin;
class PMessage;

/*!
  \ingroup buildblock
  \brief An (abstract base) class that contains information on the 
  projection data.

*/
class ProjDataInfo
{
  
public:

  /********** static members **************/

  //! Ask for the details and return a ProjDataInfo pointer
  static ProjDataInfo* 
  ask_parameters();

  //! Construct a ProjDataInfo suitable for GE Advance data
  static ProjDataInfo*  
  ProjDataInfoGE(const shared_ptr<Scanner>& scanner_ptr, 
		 const int max_delta,
		 const int num_views, const int num_tangential_poss, 
                 const bool arc_corrected = true);

  //! Construct a ProjDataInfo suitable for CTI data
  /*! \c span is used to denote the amount of axial compression (see CTI doc).
     It has to be an odd number. 
     */
  static ProjDataInfo* 
    ProjDataInfoCTI(const shared_ptr<Scanner>& scanner_ptr,
		  const int span, const int max_delta,
                  const int num_views, const int num_tangential_poss, 
                  const bool arc_corrected = true);
  
  
  /************ constructors ***********/
  // TODO should probably be protected

  //! Construct an empty object
   ProjDataInfo();
  
  //! Constructor setting all relevant info for a ProjDataInfo
   /*! The num_axial_pos_per_segment argument should be such that
       num_axial_pos_per_segment[segment_num] gives you the appropriate value 
       for a particular segment_num
       */
  ProjDataInfo(const shared_ptr<Scanner>& scanner_ptr,
		      const VectorWithOffset<int>& num_axial_pos_per_segment, 
		      const int num_views, 
		      const int num_tangential_poss);


  //! Standard trick for a 'virtual copy-constructor' 
  virtual ProjDataInfo* clone() const = 0;

  //! Destructor
  virtual ~ProjDataInfo() {}

  /**************** member functions *********/

  //  ProjDataInfo& operator=(const ProjDataInfo&);

  //! Set a new range of segment numbers
  /*! \warning the new range has to be 'smaller' than the old one. */
  void reduce_segment_range(const int min_segment_num, const int max_segment_num);
  //! Set number of views 
  void set_num_views(const int num_views);
  //! Set number of tangential positions
  void set_num_tangential_poss(const int num_tang_poss);
  //! Set number of axial positions per segment
  void set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_pos_per_segment); 

  //! Set minimum axial position number for 1 segment
  void set_min_axial_pos_num(int min_ax_pos_num, const int segment_num);
  //! Set maximum axial position number for 1 segment
  void set_max_axial_pos_num(int min_ax_pos_num, const int segment_num);
  
  //! Set minimum tangential position number
  void set_min_tangential_pos_num(int min_tang_poss);
  //! Set maximum tangential position number
  void set_max_tangential_pos_num(int max_tang_poss);
  
  //! Get number of segments
  inline int get_num_segments() const;
  //! Get number of axial positions per segment
  inline int get_num_axial_poss(const int segment_num) const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;
  //! Get minimum segment number
  inline int get_min_segment_num() const;
  //! Get maximum segment number
  inline int get_max_segment_num() const;
  //! Get minimum axial position per segmnet
  inline int get_min_axial_pos_num(const int segment_num) const;
  //! Get maximum axial position per segment
  inline int get_max_axial_pos_num(const int segment_num) const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get minimum tangential position number
  inline int get_min_tangential_pos_num() const;
  //! Get maximum tangential position number
  inline int get_max_tangential_pos_num() const;

  //! Get tangent of the co-polar angle of the normal to the projection plane
  /*! theta=0 for 'direct' planes (i.e. projection planes parallel to the scanner axis) */
  virtual float get_tantheta(const Bin&) const =0;
  
  //! Get azimuthal angle phi of the normal to the projection plane
  /*! phi=0 when the normal vector has no component along the horizontal axis */
  virtual float get_phi(const Bin&) const =0;
  
  //! Get value of the (roughly) axial coordinate in the projection plane (in mm)
  /*! t-axis is defined to be orthogonal to the s-axis (and to the vector
      normal to the projection plane */
  virtual float get_t(const Bin&) const =0;

  //! Get value of the tangential coordinate in the projection plane (in mm)
  /*! s-axis is defined to be orthogonal to the scanner axis (and to the vector
      normal to the projection plane */
  virtual float get_s(const Bin&) const =0;

  //! Get sampling distance in the \c t coordinate
  /*! For some coordinate systems, this might depend on the Bin. The 
      default implementation computes it as 
      \code
      1/2(get_t(..., ax_pos+1,...)-get_t(..., ax_pos-1,...)))
      \endcode
  */
  virtual float get_sampling_in_t(const Bin&) const;

  //! Get sampling distance in the \c s coordinate
  /*! For some coordinate systems, this might depend on the Bin. The 
      default implementation computes it as 
      \code
      1/2(get_s(..., tang_pos+1)-get_t(..., tang_pos_pos-1)))
      \endcode
  */
  virtual float get_sampling_in_s(const Bin&) const;

  //! check equality
  virtual bool operator ==(const ProjDataInfo& proj) const; 
  
  inline bool operator !=(const ProjDataInfo& proj) const; 

  //! Get empty viewgram
  Viewgram<float> get_empty_viewgram(const int view_num, const int segment_num, 
    const bool make_num_tangential_poss_odd = false) const;
  
  //! Get empty_sinogram
  Sinogram<float> get_empty_sinogram(const int ax_pos_num, const int segment_num,
    const bool make_num_tangential_poss_odd = false) const;

  //! Get empty segment sino
  SegmentByView<float> get_empty_segment_by_view(const int segment_num, 
		  	   const bool make_num_tangential_poss_odd = false) const;
  //! Get empty segment view
  SegmentBySinogram<float> get_empty_segment_by_sinogram(const int segment_num, 
				   const bool make_num_tangential_poss_odd = false) const;


  //! Get empty related viewgrams, where the symmetries_ptr specifies the symmetries to use
  RelatedViewgrams<float> get_empty_related_viewgrams(const ViewSegmentNumbers&,
    const shared_ptr<DataSymmetriesForViewSegmentNumbers>&,
    const bool make_num_tangential_poss_odd = false) const;   

  //! Get scanner pointer  
  inline const Scanner* get_scanner_ptr() const;
  
  //! Return a string describing the object
  virtual string parameter_info() const;
  
private:
  shared_ptr<Scanner> scanner_ptr;
  int min_view_num;
  int max_view_num;
  int min_tangential_pos_num;
  int max_tangential_pos_num;
  VectorWithOffset<int> min_axial_pos_per_seg; 
  VectorWithOffset<int> max_axial_pos_per_seg;
  
};

END_NAMESPACE_TOMO

#include "ProjDataInfo.inl"

#endif //  __ProjDataInfo_H__

