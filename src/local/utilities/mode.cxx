//
//
/*
  \file
  \ingroup utilities
  \author Kris Thielemans
  \brief Program that computes the mode of a set binary data.

*/
/*
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/
#include <fstream>
#include <strstream>
#include <vector>
#include <stdlib.h>


#include <algorithm>

using namespace std;

  // utility function to output an vector to a stream
template <typename elemT>
ostream& 
operator<<(ostream& str, const vector<elemT>& v)
{
      str << '{';
      for (int i=0; i<v.size()-1; i++)
	str << (unsigned int)v[i] << ", ";
      if (v.size()>0)
	str << (unsigned int)v[v.size()-1];
      str << '}' << endl;
      return str;
}

typedef unsigned char elemT;

//! computes modes in a vector
/*! 
  The mode of a vector is defined as the element that occurs
  most frequently. It is not well-defined when more than 1
  element occurs with this same frequency.
  This routine returns a vector with all such elements.
*/
template <typename elemT>
vector<elemT> compute_modes(vector<elemT>& v)
{

  //sort the elements
  sort(v.begin(), v.end());

  //cerr << v;

  vector<elemT> current_modes;
  int current_mode_count = 0;
  for (int i=0; i<v.size(); )
    {
      const elemT current_elem = v[i];
      int current_elem_count = 1;
      ++i;
      //walk through the rest of the array, till you find a different element
      while (i<v.size() && v[i] == current_elem)
	{
	  ++current_elem_count;
	  ++i;
	}
      // store mode
      if (current_elem_count > current_mode_count)
	{
	  current_modes.clear();
	  current_modes.push_back(current_elem);
	  current_mode_count = current_elem_count;
	}
      else if (current_elem_count == current_mode_count)
	{
	  current_modes.push_back(current_elem);
	}
    }

  //cerr << "modes " <<current_modes;
  return current_modes;
}



int main(int argc, char **argv)
{
  if (argc==1)
    {
      cerr << "Needs at least 1 command line parameter\n";
      cerr << "Usage: " << argv[0] << " output_filename input_filename1 input_filename2 ...\n";
      return EXIT_FAILURE;
    }
  // skip argv[0] which is the name of the current program
  --argc; ++argv;


#ifndef TESTING

  // open output file
  ofstream output(argv[0], ios::out | ios::binary); 
  if (!output)
    { 
      cerr << "Error opening output file " << argv[0] << endl;
      return EXIT_FAILURE;
    }

  // skip argv[0] which is the name of the output file
  --argc; ++argv;
  vector <ifstream *> input_files(argc);


  // open input files
  while (argc>0)
  {
    --argc;
    input_files[argc] = new ifstream(*argv, ios::in | ios::binary); 
  if (!input_files[argc])
    { 
      cerr << "Error opening input file " << *argv << endl;
      return EXIT_FAILURE;
    }
  ++argv;
  }

  vector<elemT> v(input_files.size());

  bool all_done = false;
  bool any_error = false;
  streamsize elem_count = 0;
  streamsize multiple_count = 0;
  while(!all_done)
    {
      // read next elements from files
      for (int i=0; i<input_files.size(); ++i)
	{
	  input_files[i]->read(&v[i], sizeof(v[i]));
	  // check if reading went ok
	  if (!*(input_files[i]))
	    {
	      // check if it was EOF for the first file. if not, issue warning.
	      if (i!=0 || !input_files[0]->eof())
		{	     
		  cerr << "Error reading file " << i+1 <<"\n.Exiting.\n";
		  any_error = true;
		}
	      // In any case, we break out of the loop
	      all_done = true;
	      break;
	    }
	}
      if (!all_done)
	{
	  vector<elemT> modes = compute_modes(v);
	  if (modes.size() != 1)
	    {
	      cerr << "\nWarning: multiple modes "<< modes <<"  at elem " << elem_count;
	      ++multiple_count;
	    }
	  // write first mode to file
	  output.write(&modes[modes.size()-1], sizeof(modes[0]));
	  // check if writing went ok
	  if (!output)
	    {
	      cerr << "\nError writing to output at elem " << elem_count;
	      all_done = true;
	      any_error = true;
	    }
	  else
	    ++elem_count;
	}
    }

  cerr << "Wrote " << elem_count << " elements to file\n";
  cerr << multiple_count << "multiples\n";

  // close input files by deleting the pointers
  for (int i=0; i<input_files.size(); ++i)
    delete input_files[i];


  return any_error ? EXIT_FAILURE : EXIT_SUCCESS;
  
	    
#else  
  vector<elemT> v(argc);

  while (argc>0)
  {
    --argc;
    istrstream s(*argv);
    s >> (v[argc]);
    argv++;
  }
  cerr << "elements " << v;

  const vector<elemT> modes = compute_modes(v);
  cerr << "Mode " << modes << endl;
  return EXIT_SUCCESS;

#endif

}
  
