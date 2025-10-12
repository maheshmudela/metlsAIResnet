#include <pybind11/pybind11.h>
#include "MpReader.h" // Your existing C++ header

namespace py = pybind11;

PYBIND11_MODULE(mp4module, m) 

{
    m.doc() = "Python bindings for the MP4Reader C++ class";

    py::class_<MP4Reader>(m, "MP4Reader")
        .def(py::init<>()) // Exposes the default constructor
        .def("read_file", &MP4Reader::read_file, "Reads an MP4 file from the specified path");
}

