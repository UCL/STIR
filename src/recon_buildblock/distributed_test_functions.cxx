//
//
/*
    Copyright (C) 2007- 2011, Hammersmith Imanet Ltd
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
  \ingroup distributable

  \brief Implementation of test functions in distributed namespace

  \author Tobias Beisel  

*/

#include "stir/recon_buildblock/distributed_test_functions.h"
#include "stir/recon_buildblock/distributed_functions.h"
#include "stir/VoxelsOnCartesianGrid.h"

namespace distributed
{
  void test_viewgram_slave(const  stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr)
  {	
    printf("\n-----Slave startet Test for sending viewgram----------\n");
	
    stir::Viewgram<float>* vg= NULL;		
    receive_and_construct_viewgram(vg, proj_data_info_ptr, 0);
		
    send_viewgram(*vg, 0);
    vg=NULL;
    delete vg;
  }
	
  void test_viewgram_master(stir::Viewgram<float> viewgram, const  stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr)
  {
    printf("\n-----Running Test for sending viewgram----------\n");
		
    send_viewgram(viewgram, 1);
		
    stir::Viewgram<float>* vg= NULL;		
    receive_and_construct_viewgram(vg, proj_data_info_ptr, 1);
		
    assert(vg!=NULL);
    assert(vg->has_same_characteristics(viewgram));
    assert(*vg==viewgram);
		
    for ( int tang_pos = viewgram.get_min_tangential_pos_num(); tang_pos  <= viewgram.get_max_tangential_pos_num() ;++tang_pos)  
      for ( int ax_pos = viewgram.get_min_axial_pos_num(); ax_pos <= viewgram.get_max_axial_pos_num() ;++ax_pos)
	{
	  assert(viewgram[ax_pos][tang_pos]==(*vg)[ax_pos][tang_pos]);
	  if (viewgram[ax_pos][tang_pos]!=(*vg)[ax_pos][tang_pos]) printf("-----Test sending viewgram failed!!!!-------------\n");
	}
    assert(vg->get_view_num()==viewgram.get_view_num());
    if (vg->get_view_num()!=viewgram.get_view_num()) printf("-----Test sending viewgram failed!!!!-------------\n");
    assert(vg->get_segment_num()==viewgram.get_segment_num());
    if (vg->get_segment_num()!=viewgram.get_segment_num()) printf("-----Test sending viewgram failed!!!!-------------\n");
		
    delete vg;
    printf("\n-----Test sending viewgram done-----------\n");
  }
	
  void test_image_estimate_master(const stir::DiscretisedDensity<3,float>*  input_image_ptr, int slave)
  {
    printf("\n-----Running Test for sending image estimate-----\n");
			
    int image_buffer_size;
		
    send_image_parameters(input_image_ptr, 99, slave);
    send_image_estimate(input_image_ptr, slave);
		
    stir::shared_ptr<stir::DiscretisedDensity<3, float> > target_image_sptr;
    receive_and_set_image_parameters(target_image_sptr, image_buffer_size, 99, slave);
		
    receive_image_values_and_fill_image_ptr(target_image_sptr, image_buffer_size, slave);

    assert(target_image_sptr->has_same_characteristics(*input_image_ptr));
    assert(*input_image_ptr==*target_image_sptr);

    stir::DiscretisedDensity<3,float>::const_full_iterator density_iter = input_image_ptr->begin_all();
    stir::DiscretisedDensity<3,float>::const_full_iterator density_end = input_image_ptr->end_all();
		
    stir::DiscretisedDensity<3,float>::full_iterator target_density_iter = target_image_sptr->begin_all();
			
    while (density_iter!= density_end)
      {
	assert(*density_iter == *target_density_iter);

	density_iter++;
	target_density_iter++;
      }
				
    printf("\n-----Test sending image estimate done-----\n");
  }
	
  void test_image_estimate_slave()
  {
    printf("\n-----Slave startet Test for sending image estimate-----\n");
		
    int buffer_size;
    stir::shared_ptr<stir::DiscretisedDensity<3, float> > received_image_estimate;

    receive_and_set_image_parameters(received_image_estimate, buffer_size, 99, 0);
    receive_image_values_and_fill_image_ptr(received_image_estimate, buffer_size, 0);

    send_image_parameters(received_image_estimate.get(), 99, 0);
    send_image_estimate(received_image_estimate.get(), 0);
  }
	
  void test_related_viewgrams_master(const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				     const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr,
				     stir::RelatedViewgrams<float>* y, int slave)
  {
    printf("\n-----Running Test for sending related viewgrams-----\n");
		
    send_related_viewgrams(y, 1);
	
    stir::RelatedViewgrams<float>* received_viewgrams=NULL;
		
    receive_and_construct_related_viewgrams(received_viewgrams, 
					    proj_data_info_ptr, 
					    symmetries_sptr, 1);
													
    assert(received_viewgrams!=NULL);
    assert(received_viewgrams->has_same_characteristics(*y));
    assert(*received_viewgrams==*y);
	
    stir::RelatedViewgrams<float>::iterator viewgrams_iter = y->begin();
    stir::RelatedViewgrams<float>::iterator viewgrams_end = y->end();
    stir::RelatedViewgrams<float>::iterator received_viewgrams_iter = received_viewgrams->begin();
		
    int pos=0;
    while (viewgrams_iter!= viewgrams_end)
      {
	for ( int tang_pos = (*viewgrams_iter).get_min_tangential_pos_num()  ;tang_pos  <= (*viewgrams_iter).get_max_tangential_pos_num() ;++tang_pos)  
	  for ( int ax_pos = (*viewgrams_iter).get_min_axial_pos_num(); ax_pos <= (*viewgrams_iter).get_max_axial_pos_num() ;++ax_pos)
	    { 
	      pos++;
	      assert(((*viewgrams_iter)[ax_pos][tang_pos])==(*received_viewgrams_iter)[ax_pos][tang_pos]);
	    }
	viewgrams_iter++;
	received_viewgrams_iter++;
      }
		
    received_viewgrams=NULL;
    delete received_viewgrams;		
    printf("\n-----Test sending related viewgrams done-----\n");
  }
	
  void test_related_viewgrams_slave(const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				    const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr
				    )
  {
    printf("\n-----Slave startet Test for sending related viewgrams-----\n");
			
    stir::RelatedViewgrams<float>* received_viewgrams=NULL;
		
    receive_and_construct_related_viewgrams(received_viewgrams, 
					    proj_data_info_ptr, 
					    symmetries_sptr, 0);
												
    assert(received_viewgrams!=NULL);
									
    send_related_viewgrams(received_viewgrams, 0);
		
    received_viewgrams=NULL;
    delete received_viewgrams;	
  }
	
  void test_parameter_info_master(const string str, int slave, char const * const text)
  {
    printf("\n-----Running Test for sending %s-----\n", text);
		
    string slave_string= receive_string(88, slave);
		
    assert(str.compare(slave_string)==0);
		
    /*printf("\nReceived Parameter Info:\n");
      cerr<<slave_string<<endl;
      printf("\n---------------------------\n");
      printf("\nOriginal Parameter Info:\n");
      cerr<<str<<endl;*/
    printf("\n-----Test sending %s done-----\n", text);
  }
	
  void test_parameter_info_slave(const string str)
  {
    printf("\n-----Slave startet Test for sending parameter_info or string-----\n");
    send_string(str, 88, 0);
  }
	
  void test_bool_value_master(bool value, int slave)
  {
    printf("\n-----Running Test for sending bool value-----\n");
		
    send_bool_value(value, 66, slave);
#ifndef NDEBUG
    bool received_bool_value= receive_bool_value(66, slave);
    assert(received_bool_value==value);
#endif	
    printf("\n-----Test sending bool value done-----\n");
  }
	
  void test_bool_value_slave()
  {
    printf("\n-----Slave startet Test for sending bool value-----\n");
    bool value = receive_bool_value(66, 0);
    send_bool_value(value, 66, 0);
  }
	
		
  void test_int_value_slave()
  {
    printf("\n-----Slave startet Test for sending int value-----\n");
    int value = receive_int_value(0);
    send_int_value(value, 0);
  }
	
  void test_int_value_master(int value, int slave)
  {
    printf("\n-----Running Test for sending int value-----\n");
		
    send_int_value(value, slave);
#ifndef NDEBUG
    int received_int_value= receive_int_value(slave);
    assert(received_int_value==value);
#endif
    printf("\n-----Test sending int value done-----\n");
  }
	
  void test_int_values_master(int slave)
  {
    printf("\n-----Running Test for sending int[] values-----\n");
		
    int dimensions[4];
    dimensions[0] = 1;
    dimensions[1] = 2;
    dimensions[2] = 3;
    dimensions[3] = 4;
		
    send_int_values(dimensions, 4, 66, slave);
    int received_int_values[4];

    MPI_Status status;
    status = receive_int_values(received_int_values, 4, 66);
		
    for (int i = 0; i<4; i++)
      {
	assert(dimensions[i]=received_int_values[i]);
      }
		
    printf("\n-----Test sending int[] values done-----\n");
  }
	
  void test_int_values_slave()
  {
    printf("\n-----Slave startet Test for sending int[] values-----\n");
    int received_int_values[4];
    MPI_Status status;
    status = receive_int_values(received_int_values, 4, 66);
    send_int_values(received_int_values, 4, 66, 0);
  }
}
