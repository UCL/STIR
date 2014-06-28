//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
#ifndef __stir_recon_buildblock_IterativeReconstruction_h__
#define __stir_recon_buildblock_IterativeReconstruction_h__

/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the stir::IterativeReconstruction class


  \author Matthew Jacobson
  \author Kris Thielemans
  \author Alexey Zverovich
  \author PARAPET project

*/
/* Modification history

   KT 10122001
   - added get_initial_data_ptr and 0 argument reconstruct()
*/
#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/shared_ptr.h"
#include "stir/DataProcessor.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"

START_NAMESPACE_STIR

/*! 
  \brief base class for iterative reconstruction objects
  \ingroup recon_buildblock

  This is the base class for all iterative reconstruction methods.
  It provides the basic iteration mechanisms. What each iteration
  does has to be implemented in a derived class.
  
  \par Parsing parameters

  \verbatim
  ; any parameters from Reconstruction<TargetT>

  ; see GeneralisedObjectiveFunction<TargetT>
  objective function type:= 

  
  number of subsets:= 1
  start at subset:= 0
  number of subiterations:= 1
  save images at subiteration intervals:= 1
  start at subiteration number:=2

  initial image := 
  enforce initial positivity condition:=1

  ; specify processing after every few subiterations, see DataProcessor<TargetT>
  inter-iteration filter subiteration interval:= 
  inter-iteration filter type := 

  ; write objective function value to stderr at certain subiterations
  ; default value of 0 means: do not write it at all.
  report_objective_function_values_interval:=0
  \endverbatim

  \todo move subset things somewhere else
  \todo all the <code>compute</code> functions should be <code>const</code>.
 */

template <class TargetT>
class IterativeReconstruction : public Reconstruction<TargetT>
{    
 private:
  typedef
    Reconstruction<TargetT>
    base_type;
public:

  //! accessor for the subiteration counter
  int get_subiteration_num() const
    {return subiteration_num;}

  //! accessor for finding the current subset number
  /*! The subset number is determined from the subiteration number.
      If randomise_subset_order is false, the subset number is found
      as follows:
      \code
      (subiteration_num+start_subset_num-1)%num_subsets
      \endcode
      When randomise subsets are used, a new random order is initialised
      before every full iteration. In this case, start_subset_num is ignored
      (as it doesn't make any sense).
  */
  int get_subset_num();

  //! Gets a pointer to the initial data
  /*! This is either read from file, or constructed by construct_target_ptr(). 
      In the latter case, its values are set to 0 or 1, depending on the value
      of IterativeReconstruction::initial_data_filename.

      \todo Dependency on explicit strings "1" or "0" in 
      IterativeReconstruction::initial_data_filename is not nice. 
      \todo should not return a 'bare' pointer.
  */
  virtual TargetT *
    get_initial_data_ptr() const;

  //! executes the reconstruction
  /*!
    Calls get_initial_data_ptr() and then
    reconstruct(shared_ptr<TargetT>const&).
    See end_of_iteration_processing() for info on saving to file.

    \return Succeeded::yes if everything was alright.
   */     
  virtual Succeeded 
    reconstruct();

  //! executes the reconstruction with \a target_data_sptr as initial value
  /*! After calling set_up(), repeatedly calls update_estimate(); end_of_iteration_processing();
      See end_of_iteration_processing() for info on saving to file.

      Final reconstruction is saved in \a target_data_sptr
  */
  virtual Succeeded 
    reconstruct(shared_ptr<TargetT > const& target_data_sptr);

  //! A utility function that creates a filename_prefix by appending the current subiteration number
  /*! Only works when no extension is present.
  */
  std::string
    make_filename_prefix_subiteration_num(const std::string& filename_prefix) const;

  //! A utility function that creates the output filename_prefix for the current subiteration number
  /*! Uses \a output_filename_prefix. Only works when no extension is present.
   */
  std::string
    make_filename_prefix_subiteration_num() const;

  /*! \name Functions to get parameters
   \warning Be careful with changing shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  GeneralisedObjectiveFunction<TargetT> const&
    get_objective_function() const;

  shared_ptr<GeneralisedObjectiveFunction<TargetT> >
    get_objective_function_sptr() const;

  //! the maximum allowed number of full iterations
  const int get_max_num_full_iterations() const;

  //! the number of ordered subsets
  const int get_num_subsets() const;

  //! the number of subiterations 
  const int get_num_subiterations() const;

  //! value with which to initialize the subiteration counter
  const int get_start_subiteration_num() const;

  //! the starting subset number
  const int get_start_subset_num() const;

  //TODO rename
  //! subiteration interval at which data will be saved
  const int get_save_interval() const;

  //! signals whether to randomise the subset order in each iteration
  const bool get_randomise_subset_order() const;

  //! inter-iteration filter
  const DataProcessor<TargetT>& get_inter_iteration_filter() const;

  //! subiteration interval at which to apply inter-iteration filters 
  const int get_inter_iteration_filter_interval() const;

  //! subiteration interval at which to report the values of the objective function
  const int get_report_objective_function_values_interval() const;
  //@}

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning Be careful with changing shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  //! The objective function that will be optimised
  void set_objective_function_sptr(const shared_ptr<GeneralisedObjectiveFunction<TargetT > >&);

  //! the maximum allowed number of full iterations
  void set_max_num_full_iterations(const int);

  //! the number of ordered subsets
  void set_num_subsets(const int);

  //! the number of subiterations 
  void set_num_subiterations(const int);

  //! value with which to initialize the subiteration counter
  void set_start_subiteration_num(const int);

  //! the starting subset number
  void set_start_subset_num(const int);

  //TODO rename
  //! subiteration interval at which data will be saved
  void set_save_interval(const int);

  //! signals whether to randomise the subset order in each iteration
  void set_randomise_subset_order(const bool);

  //! inter-iteration filter
  void set_inter_iteration_filter_ptr(const shared_ptr<DataProcessor<TargetT> >&);

  //! subiteration interval at which to apply inter-iteration filters 
  void set_inter_iteration_filter_interval(const int);

  //! subiteration interval at which to report the values of the objective function
  void set_report_objective_function_values_interval(const int);
  //@}

protected:
 
  IterativeReconstruction();

  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_ptr);

  //! the principal operations for updating the data iterates at each iteration
  virtual void update_estimate(TargetT &current_estimate)=0;

  //! operations for the end of the iteration
  /*! At specific subiteration numbers, this 
      <ul>
      <li>applies the inter-filtering and/or post-filtering data processor,</li>
      <li>writes the current data to file at the designated subiteration numbers 
      (including the final one). Filenames used are determined by
      Reconstruction::output_filename_prefix,</li>
      <li>writes the objective function values (using 
      GeneralisedObjectiveFunction::report_objective_function_values) to stderr.</li>
      </ul>
      If your derived class redefines this virtual function, you will
      probably want to call 
      IterativeReconstruction::end_of_iteration_processing() in there anyway.
  */
  // KT 14/12/2001 remove =0 as it's not a pure virtual and the default implementation is usually fine.
  virtual void end_of_iteration_processing(TargetT &current_estimate);

  shared_ptr<GeneralisedObjectiveFunction<TargetT > >
    objective_function_sptr;

  //! the subiteration counter
  int subiteration_num;

  //! used to abort the loop over iterations
  bool terminate_iterations;


  // parameters
 protected:
  //! the maximum allowed number of full iterations
  int max_num_full_iterations;

  //! the number of ordered subsets
  int num_subsets;

  //! the number of subiterations 
  int num_subiterations;

  //! value with which to initialize the subiteration counter
  int start_subiteration_num;

  //! name of the file containing the data for intializing the reconstruction
  std::string initial_data_filename;

  //! the starting subset number
  int start_subset_num;

  //TODO rename

  //! subiteration interval at which data will be saved
  int save_interval;

  //! signals whether to randomise the subset order in each iteration
  bool randomise_subset_order;


  //! inter-iteration filter
  shared_ptr<DataProcessor<TargetT> > inter_iteration_filter_ptr;



  
  //! subiteration interval at which to apply inter-iteration filters 
  int inter_iteration_filter_interval;
  
  //! subiteration interval at which to report the values of the objective function
  /*! \warning This is generally time-consuming.
   */
  int report_objective_function_values_interval;

  //! prompts the user to enter parameter values manually
  virtual void ask_parameters();

  virtual void set_defaults();
  virtual void initialise_keymap();
  //! used to check acceptable parameter ranges, etc...
  virtual bool post_processing();

 private:
  //! member storing the order in which the subsets will be traversed in this iteration
  /*! Initialised and used by get_subset_num() */
  VectorWithOffset<int> _current_subset_array;
  //! used to randomly generate a subset sequence order for the current iteration
  VectorWithOffset<int> randomly_permute_subset_order() const;


};

END_NAMESPACE_STIR

#endif
// __IterativeReconstruction_h__


