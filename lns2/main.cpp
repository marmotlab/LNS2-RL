#include "./inc/mylns2.h"
#include "./pybind11/include/pybind11/pybind11.h"
#include "mylns2.cpp"
#include "common.cpp"
#include "ConstraintTable.cpp"
#include "Instance.cpp"
#include "PathTable.cpp"
#include "ReservationTable.cpp"
#include "SingleAgentSolver.cpp"
#include "SIPP.cpp"
// to convert C++ STL containers to python list
#include "./pybind11/include/pybind11/stl.h"
#include "string"

namespace py=pybind11;
using namespace std;

PYBIND11_MODULE(my_lns2, m) {
    py::class_<MyLns2>(m, "MyLns2")
            .def(py::init<int, vector<vector<int>>,vector<pair<int,int>>,vector<pair<int,int>>,int,int>())
            .def("init_pp", &MyLns2::init_pp)
            .def("calculate_sipps", &MyLns2::calculate_sipps)
            .def("single_sipp", &MyLns2::single_sipp)
            .def_readwrite("vector_path", &MyLns2::vector_path)
            .def_readwrite("sipps_path", &MyLns2::sipps_path)
            .def_readwrite("makespan", &MyLns2::makespan);
}
