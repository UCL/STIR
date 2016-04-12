/*-*- C -*-*/

/**********************************************************************/

/* For numpy versions prior to 1.0, the names of certain data types
 * are different than in later versions.  This fragment provides macro
 * substitutions that allow us to support old and new versions of
 * numpy.
 */

/**********************************************************************/

/* Override the SWIG_AsVal_frag(long) fragment so that it also checks
 * for numpy scalar array types.  The code through the %#endif is
 * essentially cut-and-paste from pyprimtype.swg
 */

%fragment(SWIG_AsVal_frag(long), "header",
	  fragment="SWIG_CanCastAsInteger",
          fragment="NumPy_Backward_Compatibility")
{
  SWIGINTERN int
  SWIG_AsVal_dec(long)(PyObject * obj, long * val)
  {
    if (PyInt_Check(obj)) {
      if (val) *val = PyInt_AsLong(obj);
      return SWIG_OK;
    } else if (PyLong_Check(obj)) {
      long v = PyLong_AsLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	PyErr_Clear();
      }
    }
%#ifdef SWIG_PYTHON_CAST_MODE
    {
      int dispatch = 0;
      long v = PyInt_AsLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_AddCast(SWIG_OK);
      } else {
	PyErr_Clear();
      }
      if (!dispatch) {
	double d;
	int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
	if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, LONG_MIN, LONG_MAX)) {
	  if (val) *val = (long)(d);
	  return res;
	}
      }
    }
%#endif
    if (PyArray_IsScalar(obj,Integer))
    {
      PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_LONG);
      PyArray_CastScalarToCtype(obj, (void*)val, descr);
      Py_DECREF(descr);
      return SWIG_AddCast(SWIG_OK);
    }
    return SWIG_TypeError;
  }
}


/* Override the SWIG_AsVal_frag(unsigned long) fragment so that it
 * also checks for numpy scalar array types.  The code through the
 * %#endif is essentially cut-and-paste from pyprimtype.swg
 */

%fragment(SWIG_AsVal_frag(unsigned long),"header",
	  fragment="SWIG_CanCastAsInteger",
          fragment="NumPy_Backward_Compatibility")
{
  SWIGINTERN int
  SWIG_AsVal_dec(unsigned long)(PyObject *obj, unsigned long *val)
  {
    if (PyInt_Check(obj)) {
      long v = PyInt_AsLong(obj);
      if (v >= 0) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	return SWIG_OverflowError;
      }
    } else if (PyLong_Check(obj)) {
      unsigned long v = PyLong_AsUnsignedLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_OK;
      } else {
	PyErr_Clear();
      }
    }
%#ifdef SWIG_PYTHON_CAST_MODE
    {
      int dispatch = 0;
      unsigned long v = PyLong_AsUnsignedLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_AddCast(SWIG_OK);
      } else {
	PyErr_Clear();
      }
      if (!dispatch) {
	double d;
	int res = SWIG_AddCast(SWIG_AsVal(double)(obj,&d));
	if (SWIG_IsOK(res) && SWIG_CanCastAsInteger(&d, 0, ULONG_MAX)) {
	  if (val) *val = (unsigned long)(d);
	  return res;
	}
      }
    }
%#endif
    if (PyArray_IsScalar(obj,Integer))
    {
      PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_ULONG);
      PyArray_CastScalarToCtype(obj, (void*)val, descr);
      Py_DECREF(descr);
      return SWIG_AddCast(SWIG_OK);
    }
    return SWIG_TypeError;
  }
}

%fragment(SWIG_AsVal_frag(double),"header",
          fragment="NumPy_Backward_Compatibility") 
{
SWIGINTERN int
SWIG_AsVal_dec(double)(PyObject *obj, double *val)
{
  int res = SWIG_TypeError;
  if (PyFloat_Check(obj)) {
    if (val) *val = PyFloat_AsDouble(obj);
    return SWIG_OK;
  } else if (PyInt_Check(obj)) {
    if (val) *val = PyInt_AsLong(obj);
    return SWIG_OK;
  } else if (PyLong_Check(obj)) {
    double v = PyLong_AsDouble(obj);
    if (!PyErr_Occurred()) {
      if (val) *val = v;
      return SWIG_OK;
    } else {
      PyErr_Clear();
    }
  }
%#ifdef SWIG_PYTHON_CAST_MODE
  {
    int dispatch = 0;
    double d = PyFloat_AsDouble(obj);
    if (!PyErr_Occurred()) {
      if (val) *val = d;
      return SWIG_AddCast(SWIG_OK);
    } else {
      PyErr_Clear();
    }
    if (!dispatch) {
      long v = PyLong_AsLong(obj);
      if (!PyErr_Occurred()) {
	if (val) *val = v;
	return SWIG_AddCast(SWIG_AddCast(SWIG_OK));
      } else {
	PyErr_Clear();
      }
    }
  }
%#endif
    if (PyArray_IsScalar(obj,Double))
    {
      PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_DOUBLE);
      PyArray_CastScalarToCtype(obj, (void*)val, descr);
      Py_DECREF(descr);
      return SWIG_AddCast(SWIG_OK);
    }
  if (PyArray_IsScalar(obj,Float))
    {
      float fval;
      PyArray_Descr * descr = PyArray_DescrNewFromType(NPY_FLOAT32);
      PyArray_CastScalarToCtype(obj, (void*)&fval, descr);
      *val=(double)fval;
      Py_DECREF(descr);
      return SWIG_AddCast(SWIG_OK);
    }
    return res;
}
}
