//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup main_programs
  \brief main() for OSMAPOSLReconstruction

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/CPUTimer.h"
#ifdef _MPI
#include "stir/recon_buildblock/DistributedWorker.h"
#include "stir/recon_buildblock/distributed_functions.h"
#endif
#include "stir/PTimer.h"



USING_NAMESPACE_STIR

#ifdef PARALLEL
int master_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
  PTimer t;
  t.Reset();
  t.Start();
   
#ifdef _MPI	
  //processor-id within parallel Communicator
  int my_rank;	
	
  //saves the name of a processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];	 
	 
  //length of the processor-name
  int namelength;	
	 
  DistributedWorker<DiscretisedDensity<3,float> > *worker;
	 
  MPI_Init(&argc, &argv) ; /*Initializes the start up for MPI*/
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) ; /*Gets the rank of the Processor*/	 
  MPI_Comm_size(MPI_COMM_WORLD, &distributed::num_processors) ; /*Finds the number of processes being used*/	 
  MPI_Get_processor_name(processor_name, &namelength);
 	 
  distributed::first_iteration = true;
 	 
  fprintf(stderr, "Process %d on %s\n", my_rank, processor_name);
	 
  //master
  if (my_rank==0)
    {
#endif 

      OSMAPOSLReconstruction<DiscretisedDensity<3,float> >
    	reconstruction_object(argc>1?argv[1]:"");

	
      //return reconstruction_object.reconstruct() == Succeeded::yes ?
      //    EXIT_SUCCESS : EXIT_FAILURE;
      if (reconstruction_object.reconstruct() == Succeeded::yes) 
	{	
	  t.Stop();
#ifdef _MPI 
	  if (distributed::total_rpc_time_slaves!=0) cout << "Total time used for RPC-processing: "<< distributed::total_rpc_time_slaves << endl;
#endif
	  cout << "Total Wall clock time: " << t.GetTime() << " seconds" << endl;
	  return EXIT_SUCCESS;
	}
      else	
	{
	  return EXIT_FAILURE;
	}
            
#ifdef _MPI
    }
  else //slaves
    {		
      //create Slave Object
      worker = new DistributedWorker<DiscretisedDensity<3,float> >;
		
      //start Slave Process:
      worker->start(my_rank);
		
      delete worker;
    }	
  MPI_Finalize(); 	
#endif
}

