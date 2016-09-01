//
//
/*
    Copyright (C) 2016, UCL

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
#ifndef __stir_PostFiltering_H__
#define __stir_PostFiltering_H__

/*!
  \file

  \brief Declaration the helper class PostFiltering
  \ingroup DataProcessor

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/ParsingObject.h"
#include "stir/DataProcessor.h"
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

template<int numT, typename elemT>
class PostFiltering : public ParsingObject
{
public:
    //! Contructor with input filename
    inline PostFiltering(const char * const par_filename);

    //! Default constructor
    inline PostFiltering();

    virtual void process_data(DiscretisedDensity< numT, elemT>& arg);

    //! Check if filter exists
    bool is_filter_null();

protected:
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

private:
    shared_ptr<DataProcessor<DiscretisedDensity< numT, elemT> > > filter_sptr;

};

template<int numT, typename elemT>
PostFiltering<numT,elemT>::PostFiltering()
{

    set_defaults();
}

template <int numT, typename elemT>
PostFiltering<numT,elemT>::PostFiltering(const char * const par_filename)
{
    set_defaults();
    if (par_filename!=0)
    {
        if (parse(par_filename)==false)
            error("Exiting\n");
    }
    else
        ask_parameters();
}

template <int numT, typename elemT>
void
PostFiltering<numT,elemT>::set_defaults()
{
    filter_sptr.reset();
}

template <int numT, typename elemT>
void
PostFiltering<numT,elemT>::initialise_keymap()
{
    parser.add_start_key("Postfilter parameters");
    parser.add_parsing_key("PostFilter type", &filter_sptr);
    parser.add_stop_key("END PostFiltering Parameters");
}

template <int numT, typename elemT>
bool
PostFiltering<numT,elemT>::post_processing()
{}

template<int numT, typename elemT>
void
PostFiltering<numT,elemT>::process_data(DiscretisedDensity< numT, elemT>& arg)
{
    filter_sptr->apply(arg);
}

template<int numT, typename elemT>
bool
PostFiltering<numT,elemT>::is_filter_null()
{
    return is_null_ptr(filter_sptr);
}

END_NAMESPACE_STIR

#endif
