//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisation

  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_BinNormalisation_H__
#define __stir_recon_buildblock_BinNormalisation_H__


#include "stir/RegisteredObject.h"
#include "stir/Bin.h"
#include "stir/shared_ptr.h"
#include "stir/deprecated.h"

START_NAMESPACE_STIR

template <typename elemT> class RelatedViewgrams;
class Succeeded;
class ProjDataInfo;
class ProjData;
class DataSymmetriesForViewSegmentNumbers;
class ExamInfo;
/*!
  \ingroup normalisation
  \brief Abstract base class for implementing bin-wise normalisation of data.

  As part of the measurement model in PET, there usually is some multiplicative 
  correction for every bin, as in 
  \f[ P^\mathrm{full}_{bv} = \mathrm{norm}_b P^\mathrm{normalised}_{bv} \f]
  This multiplicative correction is usually split in the \c normalisation 
  factors (which are scanner dependent) and the \c attenuation factors (which 
  are object dependent). 

  The present class can be used for both of these factors.
*/
class BinNormalisation : public RegisteredObject<BinNormalisation>
{
public:

  BinNormalisation();

  virtual ~BinNormalisation();
  virtual float get_calibration_factor() const {return -1;}

  //! check if we would be multiplying with 1 (i.e. do nothing)
  /*! This function can be used to check if the operations are guaranteed to do nothing
      (while potentially taking time and effort). The base-class sets this to always
      return false. It is up to the derived class to change this.
  */
  virtual inline bool is_trivial() const { return false;}

  //! initialises the object and checks if it can handle such projection data
  /*! Default version does nothing. */
  virtual Succeeded set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,const shared_ptr<const ProjDataInfo>&);

  //! Return the 'efficiency' factor for a single bin
  /*! With the notation of the class documentation, this returns the factor
    \f$\mathrm{norm}_b \f$. 

    \warning Some derived classes might implement this very inefficiently.
  */
  virtual float get_bin_efficiency(const Bin& bin) const =0;

  //! normalise some data
  /*! 
    This would be used for instance to precorrect unnormalised data. With the
    notation of the class documentation, this would \c divide by the factors 
    \f$\mathrm{norm}_b \f$.

    Default implementation divides with the factors returned by get_bin_efficiency()
    (after applying a threshold to avoid division by 0).
  */
  virtual void apply(RelatedViewgrams<float>&) const;

  //! undo the normalisation of some data
  /*! 
    This would be used for instance to bring geometrically forward projected data to 
    the mean of the measured data. With the
    notation of the class documentation, this would \c multiply by the factors 
    \f$\mathrm{norm}_b \f$.

    Default implementation multiplies with the factors returned by get_bin_efficiency().
  */
  virtual void undo(RelatedViewgrams<float>&) const; 

  //! normalise some data
  /*! 
    This would be used for instance to precorrect unnormalised data. With the
    notation of the class documentation, this would \c divide by the factors 
    \f$\mathrm{norm}_b \f$.

    This just loops over all RelatedViewgrams. 

    The default value for the symmetries means that TrivialDataSymmetriesForBins will be used.
  */
  void apply(ProjData&, 
             shared_ptr<DataSymmetriesForViewSegmentNumbers> = shared_ptr<DataSymmetriesForViewSegmentNumbers>()) const;

  //! undo the normalisation of some data
  /*! 
    This would be used for instance to bring geometrically forward projected data to 
    the mean of the measured data. With the
    notation of the class documentation, this would \c multiply by the factors 
    \f$\mathrm{norm}_b \f$.

    This just loops over all RelatedViewgrams. 

    The default value for the symmetries means that TrivialDataSymmetriesForBins will be used.
  */
  void undo(ProjData&, 
            shared_ptr<DataSymmetriesForViewSegmentNumbers> = shared_ptr<DataSymmetriesForViewSegmentNumbers>()) const; 

  //! old interface. do not use
  STIR_DEPRECATED void undo(ProjData& p,const double /*start_time*/, const double /*end_time*/, 
            shared_ptr<DataSymmetriesForViewSegmentNumbers> sym = shared_ptr<DataSymmetriesForViewSegmentNumbers>()) const
  {
    this->undo(p, sym);
  }

  
  //! old interface. do not use
  STIR_DEPRECATED void apply(ProjData& p,const double /*start_time*/, const double /*end_time*/, 
            shared_ptr<DataSymmetriesForViewSegmentNumbers> sym = shared_ptr<DataSymmetriesForViewSegmentNumbers>()) const
  {
    this->apply(p, sym);
  }
  
  void set_exam_info_sptr(const shared_ptr<const ExamInfo> _exam_info_sptr);

  shared_ptr<const ExamInfo> get_exam_info_sptr() const ;

 protected:
  //! check if the argument is the same as what was used for set_up()
  /*! calls error() if anything is wrong.

      If overriding this function in a derived class, you need to call this one.
   */
  virtual void check(const ProjDataInfo& proj_data_info) const;
  
  virtual void check(const ExamInfo& exam_info) const;
  bool _already_set_up;
private:
  shared_ptr<const ExamInfo> exam_info_sptr;
  shared_ptr<const ProjDataInfo> _proj_data_info_sptr;
};

END_NAMESPACE_STIR

#endif
