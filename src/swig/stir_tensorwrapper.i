/*
    SWIG interface file for TensorWrapper<3, float>
    Provides Python bindings for TensorWrapper<3, float> with PyTorch tensor support.
*/

// %module stir_tensorwrapper

%{

#include <torch/torch.h>
#include "stir/TensorWrapper.h"
#include <torch/csrc/autograd/python_variable.h>
%}

%include "stir/TensorWrapper.h"

%shared_ptr(stir::TensorWrapper<3, float>);

namespace stir
{
// Explicitly instantiate TensorWrapper<3, float>
%template(TensorWrapper3DFloat) TensorWrapper<3, float>;

// %extend stir::TensorWrapper<3, float> {
//     // Convert TensorWrapper to a PyTorch tensor
//     PyObject* to_python_tensor() {
//         return THPVariable_Wrap(self->getTensor());
//     }

//     // Set a PyTorch tensor into TensorWrapper
//     void from_python_tensor(PyObject* py_tensor) {
//         if (!THPVariable_Check(py_tensor)) {
//             // SWIG_exception(SWIG_TypeError, "Expected a PyTorch tensor");
//         }
//         auto tensor = THPVariable_Unpack(py_tensor);
//         self->getTensor() = tensor;
//     }

//     // Constructor to create TensorWrapper from a PyTorch tensor
//     TensorWrapper<3, float>(PyObject* py_tensor, const std::string& device = "cpu") {
//         if (!THPVariable_Check(py_tensor)) {
//             // SWIG_exception(SWIG_TypeError, "Expected a PyTorch tensor");
//         }
//         auto tensor = THPVariable_Unpack(py_tensor);
//         return new stir::TensorWrapper<3, float>(tensor, device);
//     }

//     // Print the device of the tensor
//     void print_device() {
//         self->print_device();
//     }

//     // Print the sizes of the tensor
//     void print_sizes() {
//         self->printSizes();
//     }
// }

}