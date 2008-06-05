//
// $Id$
//
/*
    Copyright (C) 2007- $Date$, Hammersmith Imanet Ltd
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

  \brief Implementation of functions in distributed namespace

  \author Tobias Beisel  

  $Date$
*/

#include "stir/recon_buildblock/distributed_functions.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/Succeeded.h"


namespace distributed
{
  //timings and tests
  bool rpc_time=false;
  bool test_send_receive_times=false;
	
  double total_rpc_time=0;
  double min_threshold=0.1;
  double total_rpc_time_slaves=0.0;
  double total_rpc_time_2=0.0;
  bool test=false;
	
	
  //global variable often used
  int num_processors;	 
  int length;
		
  int processor;
  bool first_iteration;
  int iteration_counter=0;
  int image_buffer_size;
  MPI_Status status;
  float parameters[6]; //array for receiving image parameters
  int sizes[6];	//array for receiving image dimensions		
   	
  stir::HighResWallClockTimer t;
   		
   	
  //--------------------------------------Send Operations-------------------------------------
	
  void send_int_value(int value, int destination)
  {		
    int i = value;

#ifdef	STIR_MPI_TIMINGS	
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (destination == -1) MPI_Bcast(&i, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    else MPI_Send(&i, 1, MPI_INT, destination, INT_TAG,  MPI_COMM_WORLD);

#ifdef STIR_MPI_TIMINGS		
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending int value took " << t.GetTime() << " seconds" << endl;
#endif
  }
	
  void send_string(string str, int tag, int destination)
  {
    //prepare sending parameter info
    length=str.length()+1;
    char * buf= new char[length];
    strcpy(buf, str.c_str());	
		
    //send parameter info 
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    if (destination == -1)
      for (int processor=1; processor<num_processors; processor++)
	{
	  MPI_Send(&length, 1, MPI_INT, processor, tag, MPI_COMM_WORLD);
	  MPI_Send(buf, length, MPI_CHAR, processor, tag, MPI_COMM_WORLD);
	}
    else 
      {
	MPI_Send(&length, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
	MPI_Send(buf, length, MPI_CHAR, destination, tag, MPI_COMM_WORLD);
      }
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending string took " << t.GetTime() << " seconds" << endl;
#endif

    delete[] buf;
  }
	
  void send_bool_value(bool value, int tag, int destination)
  {
    int i;
    if (value==true) i=1;
    else i=0; 
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    if (destination==-1||tag ==-1) MPI_Bcast(&i, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    else MPI_Send(&i, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending bool value took " << t.GetTime() << " seconds" << endl;
#endif

  }
	
  void send_int_values(int * values, int count, int tag, int destination)
  {
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (destination == -1)
      {	
	for (processor=1; processor<num_processors; processor++)
	  MPI_Send(values, count, MPI_INT, processor, tag, MPI_COMM_WORLD);
      }
    else MPI_Send(values, count, MPI_INT, destination, tag, MPI_COMM_WORLD);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending int values took " << t.GetTime() << " seconds"<< endl;
#endif

  }
	
  void send_double_values(double * values, int count, int tag, int destination)
  {

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (destination == -1)
      {	
	for (processor=1; processor<num_processors; processor++)
	  MPI_Send(values, count, MPI_DOUBLE, processor, tag, MPI_COMM_WORLD);
      }
    else MPI_Send(values, count, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master: sending double values took " << t.GetTime() << " seconds" << endl;
#endif
  }
	
  void send_image_parameters(const stir::DiscretisedDensity<3,float>* input_image_ptr, int tag, int destination)
  {
    //cast to allow getting image dimensions and grid_spacing
    const stir::VoxelsOnCartesianGrid<float>* image =
      dynamic_cast<const stir::VoxelsOnCartesianGrid<float>* >(input_image_ptr);
			  
    //input_image dimensions
    int * sizes = new int[6];
    sizes[0] = image->get_min_x();
    sizes[1] = image->get_min_y();
    sizes[2] = image->get_min_z();
    sizes[3] = image->get_max_x();
    sizes[4] = image->get_max_y();
    sizes[5] = image->get_max_z();
		
    //buffer for input_image-array
    image_buffer_size=(sizes[3]-sizes[0]+1)*(sizes[4]-sizes[1]+1)*(sizes[5]-sizes[2]+1);
		
    //get origin of input_image_ptr
    stir::CartesianCoordinate3D<float> origin = input_image_ptr->get_origin();
		
    //get grid_spacing of input_image_ptr
    stir::CartesianCoordinate3D<float> grid_spacing = image->get_grid_spacing();
		
    float * parameters = new float[6];
    parameters[0] = origin.x();
    parameters[1] = origin.y();
    parameters[2] = origin.z();
    parameters[3] = grid_spacing.x();
    parameters[4] = grid_spacing.y();
    parameters[5] = grid_spacing.z();
		
		
    //Sending image parameters
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    if (tag == -1 || destination == -1) MPI_Bcast(parameters, 6, MPI_FLOAT, 0,  MPI_COMM_WORLD);
    else MPI_Send(parameters, 6, MPI_FLOAT, destination, tag, MPI_COMM_WORLD);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending image parameters took " << t.GetTime() << " seconds" << endl;
#endif
		
    //send dimensions to construct IndexRange-object
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    if (tag == -1 || destination == -1) MPI_Bcast(sizes, 6, MPI_INT, 0,  MPI_COMM_WORLD);
    else MPI_Send(sizes, 6, MPI_INT, destination, tag, MPI_COMM_WORLD);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending image dimensions took " << t.GetTime() << " seconds" << endl;
#endif
		
    delete[] sizes;
    delete[] parameters;
  }
	
  void send_image_estimate(const stir::DiscretisedDensity<3,float>* input_image_ptr, int destination)
  {		
    float *image_buf = new float[image_buffer_size];
	
    //serialize input_image into 1-demnsional array
    stir::DiscretisedDensity<3,float>::full_iterator density_iter = const_cast<stir::DiscretisedDensity<3,float>* >(input_image_ptr)->begin_all();
    stir::DiscretisedDensity<3,float>::full_iterator density_end = const_cast<stir::DiscretisedDensity<3,float>* >(input_image_ptr)->end_all();
		
    int pos=0;
    while (density_iter!= density_end)
      {
	image_buf[pos]=*density_iter;
	density_iter++;
	pos++;
      }
		
    //send input image
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (destination == -1)
      for (int processor=1; processor<num_processors; processor++)
	MPI_Send(image_buf, image_buffer_size, MPI_FLOAT, processor, IMAGE_ESTIMATE_TAG, MPI_COMM_WORLD);
    else MPI_Send(image_buf, image_buffer_size, MPI_FLOAT, destination, IMAGE_ESTIMATE_TAG, MPI_COMM_WORLD);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: sending image values took " << t.GetTime() << " seconds" << endl;
#endif
  }
	
  void send_proj_data_info(stir::ProjDataInfo* data, int destination)
  {
    stir::ProjDataInterfile projection_data_for_slave(data,"for_slave");
				
    std::ifstream is;
    is.open ("for_slave.hs", ios::binary );
		
    // get length of file:
    is.seekg (0, ios::end);
    length = is.tellg();
    is.seekg (0, ios::beg);
		
    length++;
	
    char * file_buffer = new char[length];
		
    // read data as a block:
    is.read (file_buffer,length-1);
    is.close();
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    //send text_buffer
    if (destination==-1)
      for (processor=1; processor<num_processors; processor++)
	{
	  MPI_Send(&length, 1, MPI_INT, processor, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD);
	  MPI_Send(file_buffer, length, MPI_CHAR, processor, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD);
	}
    else 
      {
	MPI_Send(&length, 1, MPI_INT, destination, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD);
	MPI_Send(file_buffer, length, MPI_CHAR, destination, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD);
      }
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master: sending projection_data_info took " << t.GetTime() << " seconds" << endl;
#endif
		
		
    if( remove( "for_slave.hs" ) != 0 )
      stir::warning( "Error deleting temporary file" );		  
    if( remove( "for_slave.s" ) != 0 )
      stir::warning( "Error deleting temporary file" );		
    		
    delete[] file_buffer;
  }	
	
  void send_related_viewgrams(stir::RelatedViewgrams<float>* viewgrams, int destination)
  {
    //broadcast count of viewgrams to be received
    int num_viewgrams=viewgrams->get_num_viewgrams();
		
    //run through viewgrams	
    stir::RelatedViewgrams<float>::iterator viewgrams_iter = viewgrams->begin();
    stir::RelatedViewgrams<float>::iterator viewgrams_end = viewgrams->end();
		
    int * viewgram_counts = new int[1];
    viewgram_counts[0] = num_viewgrams;
	
    send_int_values(viewgram_counts, 1, VIEWGRAM_COUNT_TAG, destination);
		
    while (viewgrams_iter!= viewgrams_end)
      {	
	send_viewgram(*viewgrams_iter, destination);
	viewgrams_iter++;
      }
  }
	
  void send_viewgram(stir::Viewgram<float> viewgram, int destination)
  {
    //send dimensions of viewgram (axial and tangential positions and the view and segment numbers)
    int * viewgram_values = new int[6];
    viewgram_values[0] = viewgram.get_min_axial_pos_num();
    viewgram_values[1] = viewgram.get_max_axial_pos_num();
    viewgram_values[2] = viewgram.get_min_tangential_pos_num();
    viewgram_values[3] = viewgram.get_max_tangential_pos_num();
    viewgram_values[4] = viewgram.get_view_num();
    viewgram_values[5] = viewgram.get_segment_num();
		
    send_int_values(viewgram_values, 6, VIEWGRAM_DIMENSIONS_TAG, destination);
		
    //allocate send-buffer
    int buffer_size=(viewgram_values[1]-viewgram_values[0]+1)*(viewgram_values[3]-viewgram_values[2]+1);
    float *viewgram_buf = new float[buffer_size];
	
    int pos=0;
    for ( int tang_pos = viewgram_values[2] ;tang_pos  <= viewgram_values[3] ;++tang_pos)  
      for ( int ax_pos = viewgram_values[0]; ax_pos <= viewgram_values[1] ;++ax_pos)
	{ 
	  viewgram_buf[pos]=viewgram[ax_pos][tang_pos];
	  pos++;
	}
	
    //send array
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    MPI_Send(viewgram_buf, buffer_size, MPI_FLOAT, destination, VIEWGRAM_TAG, MPI_COMM_WORLD);			

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master: sending viewgram took " << t.GetTime() << " seconds" << endl;
#endif
		
    delete[] viewgram_buf;
    delete[] viewgram_values;
  }
	
  //--------------------------------------Receive Operations-------------------------------------
	
  int receive_int_value(int source)
  {	
    int i;
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    if (source==-1) MPI_Bcast(&i, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    else MPI_Recv(&i, 1, MPI_INT, source, INT_TAG, MPI_COMM_WORLD, &status);
	
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received int value after " << t.GetTime() << " seconds" << endl;
#endif
	
    return i;
  }
	
  string receive_string(int tag, int source)
  {		
    //Receive projector-pair registered_keyword
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    MPI_Recv(&length, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    char * buf= new char[length];
    MPI_Recv(buf, length, MPI_CHAR, source, tag, MPI_COMM_WORLD, & status);
	
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received string value after " << t.GetTime() << " seconds" << endl;
#endif
		
    //convert to string
    string str(buf);
    delete[] buf;
    return str;
  }
	
  void receive_and_initialize_projectors(stir::shared_ptr<stir::ProjectorByBinPair> &projector_pair_ptr, int source)
  {			
    //Receive projector-pair registered_keyword
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    MPI_Recv(&length, 1, MPI_INT, source, REGISTERED_NAME_TAG, MPI_COMM_WORLD, &status);
    char * buf= new char[length];
    MPI_Recv(buf, length, MPI_CHAR, source, REGISTERED_NAME_TAG, MPI_COMM_WORLD, & status);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received REGISTERED_NAME_TAG value after " << t.GetTime() << " seconds" << endl;
#endif
		
    //convert to string
    string registered_name_proj_pair(buf);
    delete[] buf;
		
    //Receive parameter info		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    MPI_Recv(&length, 1, MPI_INT, source, PARAMETER_INFO_TAG, MPI_COMM_WORLD, &status);

    char * buf3= new char[length];	
    MPI_Recv(buf3, length, MPI_CHAR, source, PARAMETER_INFO_TAG, MPI_COMM_WORLD, & status);

#ifdef STIR_MPI_TIMINGS		
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received parameter info value after " << t.GetTime() << " seconds" << endl;
#endif
		
    //convert to string 
    string parameter_info(buf3);
    delete[] buf3;
		
    //construct new Backprojector and Forward Projector, projector_pair_ptr
    std::istringstream parameter_info_stream(parameter_info);
		
    projector_pair_ptr = stir::RegisteredObject<stir::ProjectorByBinPair>::
      read_registered_object(&parameter_info_stream, registered_name_proj_pair);
  }
	
  bool receive_bool_value(int tag, int source)
  {
    int * i=new int[sizeof(int)];
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (tag==-1 || source == -1) MPI_Bcast(i, 1, MPI_INT, 0,  MPI_COMM_WORLD);
    MPI_Recv(i, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received bool value after " << t.GetTime() << " seconds" << endl;
#endif
		
    if (*i==1) return true;
    else return false;		
  }
	
  MPI_Status receive_int_values(int * values, int count, int tag)
  {

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    if (tag == ARBITRARY_TAG) MPI_Recv(values, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    else MPI_Recv(values, count, MPI_INT, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master/Slave: received int values after " << t.GetTime() << " seconds"<< endl;
#endif

    return status;
  }
	
  MPI_Status receive_double_values(double * values, int count, int tag)
  {

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    if (tag == ARBITRARY_TAG) MPI_Recv(values, count, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    else MPI_Recv(values, count, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
		
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received double values after " << t.GetTime() << " seconds" << endl;
#endif
		
    return status;
  }
	
  void receive_and_set_image_parameters(stir::shared_ptr<stir::DiscretisedDensity<3, float> > &image_ptr, int &buffer, int tag, int source)
  {
    //receive image parameters (origin and grid_spacing)
#ifdef STIR_MPI_TIMINGS   		
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    if (tag==-1) MPI_Bcast(parameters, 6, MPI_FLOAT, source,  MPI_COMM_WORLD);
    else MPI_Recv(parameters, 6, MPI_FLOAT, source, tag,  MPI_COMM_WORLD, &status);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received origin and grid_spacing after " << t.GetTime() << " seconds" << endl;
		
    //receive dimensions of Image-data
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    if (tag==-1) MPI_Bcast(sizes, 6, MPI_INT, source,  MPI_COMM_WORLD);
    else MPI_Recv(sizes, 6, MPI_INT, source, tag,  MPI_COMM_WORLD, &status);

#ifdef STIR_MPI_TIMINGS   	
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received image dimensions after " << t.GetTime() << " seconds" << endl;
#endif
	
    buffer=(sizes[3]-sizes[0]+1)*(sizes[4]-sizes[1]+1)*(sizes[5]-sizes[2]+1);
   		
    //construct new index range from received values		
    stir::IndexRange<3> 
      range(stir::CartesianCoordinate3D<int>(sizes[2],sizes[1],sizes[0]),
	    stir::CartesianCoordinate3D<int>(sizes[5],sizes[4],sizes[3]));
          
    //construct new image object to save received input-image values
    stir::CartesianCoordinate3D<float> origin( parameters[2],  parameters[1],  parameters[0]);
    stir::CartesianCoordinate3D<float> grid_spacing(parameters[5],  parameters[4],  parameters[3]);
        
    stir::VoxelsOnCartesianGrid<float> *voxels = new stir::VoxelsOnCartesianGrid<float>(range, origin, grid_spacing);
    stir::shared_ptr<stir::VoxelsOnCartesianGrid<float> > tmpPtr(voxels);
        
    //point pointer to newly created image_estimate
    image_ptr = *(stir::shared_ptr<stir::DiscretisedDensity<3, float> >*)&tmpPtr;  		
  }
   
  MPI_Status receive_image_values_and_fill_image_ptr(stir::shared_ptr<stir::DiscretisedDensity<3, float> > &image_ptr, int buffer_size, int source)
  {
    //buffer for input_image
    float * buffer = new float[buffer_size]; 

#ifdef STIR_MPI_TIMINGS   		
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
		
    MPI_Recv(buffer, buffer_size, MPI_FLOAT, source, IMAGE_ESTIMATE_TAG, MPI_COMM_WORLD, & status);

#ifdef STIR_MPI_TIMINGS		
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received image values after " << t.GetTime() << " seconds" << endl;
#endif
		
    if (status.MPI_TAG==END_RECONSTRUCTION_TAG) 
      {
	delete[] buffer;
	return status;
      }
    else 
      {
	stir::DiscretisedDensity<3,float>::full_iterator density_iter = image_ptr->begin_all();
	stir::DiscretisedDensity<3,float>::full_iterator density_end = image_ptr->end_all();
			 
	int pos=0;
	while (density_iter!= density_end)
	  {
	    *density_iter = buffer[pos];
	    pos++;
	    density_iter++;
	  }
	delete[] buffer;
      }
    return status;  		   	
  }
	
  void receive_and_construct_proj_data_info_ptr(stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_sptr, int source)
  {
    int len;
    //receive projection data info pointer
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif
	
    MPI_Recv(&len, 1, MPI_INT, source, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD, &status);
    char * proj_data_info_buf= new char[len];
    MPI_Recv(proj_data_info_buf, len, MPI_CHAR, source, PROJECTION_DATA_INFO_TAG, MPI_COMM_WORLD, & status);

#ifdef STIR_MPI_TIMINGS	
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received proj_data_info after " << t.GetTime() << " seconds" << endl;
#endif
	
    //construct pojector_info_ptr
    std::istringstream pojector_info_ptr_stream(proj_data_info_buf);
    stir::InterfilePDFSHeader hdr;  
    hdr.parse(pojector_info_ptr_stream);
	
    stir::shared_ptr<stir::ProjDataInfo> temp_proj_data_info_sptr(hdr.data_info_ptr);
		
    proj_data_info_sptr = temp_proj_data_info_sptr;
  }
   
  void receive_and_construct_related_viewgrams(stir::RelatedViewgrams<float>*& viewgrams, 
					       const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
					       const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr,
					       int source)
  {
    //receive count of viewgrams
    int * int_values = new int[1];
    status=distributed::receive_int_values(int_values, 1, VIEWGRAM_COUNT_TAG);
   	
    vector<stir::Viewgram<float> > viewgrams_vector;
   		
    viewgrams_vector.reserve(int_values[0]);
   	
    for (int i=0; i<int_values[0]; i++) 
      {	
	stir::Viewgram<float>* vg = NULL;
   			
	//receive a viewgram from Master
	receive_and_construct_viewgram(vg, proj_data_info_ptr, source);
	   		    			
	//add viewgram to Viewgram-vector
	viewgrams_vector.push_back(*vg);	
	delete vg;	    
      }
   		
    //use viewgram-vector and symmetries-pointer to construct related viewgrams element
    stir::RelatedViewgrams<float>* tmp_viewgrams = new stir::RelatedViewgrams<float>(viewgrams_vector, symmetries_sptr);

    delete[] int_values;

    //point viewgrams to newly constructed related-viewgrams
    viewgrams= tmp_viewgrams;
  }
   
  void receive_and_construct_viewgram(stir::Viewgram<float>*& viewgram, 
				      const  stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				      int source)
  {
    //receive dimension of viewgram (vlues 0-3) and view_num + segment_num (values 4-5)
    int * viewgram_values = new int[6];
   		
    status = receive_int_values(viewgram_values, 6, VIEWGRAM_DIMENSIONS_TAG);
		
    int v_num = viewgram_values[4];
    int s_num = viewgram_values[5];
		
    stir::Viewgram<float>* tmp_viewgram= new stir::Viewgram<float>(proj_data_info_ptr, v_num, s_num);
		
    //allocate receive-buffer
    int buffer_size=(viewgram_values[1]-viewgram_values[0]+1)*(viewgram_values[3]-viewgram_values[2]+1);
    float *viewgram_buf = new float[buffer_size];
		
    //receive viewgram array

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    MPI_Recv(viewgram_buf, buffer_size, MPI_FLOAT, source, VIEWGRAM_TAG, MPI_COMM_WORLD, &status);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: received viewgram_array after " << t.GetTime() << " seconds" << endl;
#endif
		
    //construct viewgram using array and dimensions
    int pos=0;
    for ( int tang_pos = viewgram_values[2] ;tang_pos  <= viewgram_values[3] ;++tang_pos)  
      for ( int ax_pos = viewgram_values[0]; ax_pos <= viewgram_values[1] ;++ax_pos)
	{ 
	  (*tmp_viewgram)[ax_pos][tang_pos]=viewgram_buf[pos];
	  pos++;
	}

    //point viegram to newly created tmp_viewgram
    viewgram=tmp_viewgram;
    delete[] viewgram_buf;
    delete[] viewgram_values;
  }	
    
  //--------------------------------------Reduce Operations-------------------------------------
	
  void reduce_received_output_image(stir::DiscretisedDensity<3,float>* output_image_ptr, int destination)
  {		
    float *output_buf = new float[image_buffer_size];
    float *image_buf = new float[image_buffer_size];
		
    //initialize output_buffer
    for (int i=0; i<image_buffer_size;i++) output_buf[i]=0.0;
			
    //receive output image values
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    MPI_Reduce(image_buf, output_buf, image_buffer_size, MPI_FLOAT, MPI_SUM, destination, MPI_COMM_WORLD);		

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Master: reduced output_image after " << t.GetTime() << " seconds" << endl;
#endif

    printf("Master: output_image reduced.\n");
		
    //serialize input_image into 1-demnsional array
    stir::DiscretisedDensity<3,float>::full_iterator density_iter = output_image_ptr->begin_all();
    stir::DiscretisedDensity<3,float>::full_iterator density_end = output_image_ptr->end_all();
		 
    int pos=0;
    while (density_iter!= density_end)
      {
	*density_iter = output_buf[pos];
	pos++;
	density_iter++;
      }
    delete[] image_buf;
    delete[] output_buf;
  }
	
  void reduce_output_image(stir::shared_ptr<stir::DiscretisedDensity<3, float> > &output_image_ptr, int image_buffer_size, int my_rank, int destination)
  {
    float *image_buf = new float[image_buffer_size];
    float *output_buf = new float[image_buffer_size];
		
    //serialize input_image into 1-demnsional array
    stir::DiscretisedDensity<3,float>::full_iterator density_iter = output_image_ptr->begin_all();
    stir::DiscretisedDensity<3,float>::full_iterator density_end = output_image_ptr->end_all();
		 
    int pos=0;
    while (density_iter!= density_end)
      {
	image_buf[pos]=*density_iter;
	pos++;
	density_iter++;
      }
		
    //reduction of output_image at master
#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) {t.Reset(); t.Start();} 
#endif

    MPI_Reduce(image_buf, output_buf, image_buffer_size, MPI_FLOAT, MPI_SUM, destination, MPI_COMM_WORLD);

#ifdef STIR_MPI_TIMINGS
    if (test_send_receive_times) t.Stop();
    if (test_send_receive_times && t.GetTime()>min_threshold) cout << "Slave: reduced output_image after " << t.GetTime() << " seconds" << endl;
#endif		
    delete[] image_buf;
    delete[] output_buf;
  }
	
}
