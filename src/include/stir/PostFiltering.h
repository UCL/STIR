//
//
/*
    Copyright (C) 2016, UCL

    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
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

    void set_filter_sptr(shared_ptr<DataProcessor< DataT > > filter_sptr);
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
