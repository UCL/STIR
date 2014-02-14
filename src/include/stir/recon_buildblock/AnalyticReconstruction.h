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
#ifndef __stir_recon_buildblock_AnalyticReconstruction_H__
#define __stir_recon_buildblock_AnalyticReconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the stir::AnalyticReconstruction class

  \author Kris Thielemans
  \author Matthew Jacobson
  \author Claire Labbe
  \author PARAPET project

*/
/* Modification history

   KT 10122001
   - added construct_target_image_ptr and 0 argument reconstruct()
*/

#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include <string>


#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR


class Succeeded;

/*!
  \brief base class for all analytic reconstruction algorithms
  \ingroup recon_buildblock
  This class provides extra functinoality (compared to Reconstruction) as it assumes
  that the \c TargetT is really a VoxelsOnCartesianGrid.

  \todo Currently the template argument uses DiscretisedDensity
  because of conversion problems with stir::shared_ptr. Maybe it will be
  possible to correct this once we use boost:shared_ptr.
  */
class AnalyticReconstruction : public Reconstruction<DiscretisedDensity<3,float> >
{
protected:
  typedef DiscretisedDensity<3,float> TargetT;
private:
  typedef Reconstruction<TargetT > base_type;
public:
  
  //! construct an image from parameters set (e.g. during parsing)
  virtual DiscretisedDensity<3,float>*  
    construct_target_image_ptr() const;

  //! reconstruct and write to file
  /*!
    Calls construct_target_image_ptr() and then actual_reconstruct(target_image_sptr).
    At the end of the reconstruction, the final image is saved to file as given in 
    Reconstruction::output_filename_prefix. 
    \return Succeeded::yes if everything was alright.
   */
  virtual Succeeded 
    reconstruct(); 

  //! executes the reconstruction storing result in \c target_image_sptr
  /*!
    Calls actual_reconstruct()
    \return Succeeded::yes if everything was alright.

   \par Developer\'s note

   Because of C++ rules, overloading one of the reconstruct() functions
   in a derived class, hides the other. So, we need an implementation for 
   this function. This is the reason to use the actual_reconstruct function.
  */     
  virtual Succeeded 
    reconstruct(shared_ptr<TargetT> const& target_image_sptr);

  // parameters
 protected:

  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy;

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; 

  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  //! offset in the z-direction
  double Zoffset;


  //! the input projection data file name
  string input_filename;
  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_ptr;
  // KT 20/06/2001 disabled
#if 0
  //! number of views to add (i.e. mashing)
  int num_views_to_add;
#endif



protected:
  //! executes the reconstruction storing result in \c target_image_sptr
  /*!
    \return Succeeded::yes if everything was alright.
  */     
  virtual Succeeded 
    actual_reconstruct(shared_ptr<TargetT> const& target_image_sptr) = 0;
 
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();  
  virtual void set_defaults();
  virtual void initialise_keymap();


};

END_NAMESPACE_STIR

    
#endif

