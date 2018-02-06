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
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <class DataT>
        class PostFiltering : public ParsingObject
{
public:

    //! Default constructor
    PostFiltering();

    virtual ~PostFiltering(){}

    Succeeded process_data(DataT& arg);

    //! Check if filter exists
    bool is_filter_null();

protected:
    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

private:
    shared_ptr<DataProcessor< DataT > > filter_sptr;

};

END_NAMESPACE_STIR
#include "stir/PostFiltering.inl"
#endif
