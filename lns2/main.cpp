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
    py::class_<Neighbor>(m, "Neighbor")
            .def_readwrite("colliding_pairs",&Neighbor::colliding_pairs);

    py::class_<MyLns2>(m, "MyLns2")
            .def(py::init<int,  vector<vector<int>>,vector<pair<int,int>>,vector<pair<int,int>>,int,int,int>())
            .def("init_pp", &MyLns2::init_pp)
            .def("select_and_sipps", &MyLns2::select_and_sipps)
            .def("rest_lns", &MyLns2::rest_lns)
            .def("extract_path", &MyLns2::extract_path)
            .def("add_sipps", &MyLns2::add_sipps)
            .def("replan_sipps", &MyLns2::replan_sipps)
            .def_readwrite("vector_path", &MyLns2::vector_path)
            .def_readwrite("sipps_path", &MyLns2::sipps_path)
            .def_readwrite("num_of_colliding_pairs", &MyLns2::num_of_colliding_pairs)
            .def_readwrite("shuffled_agents", &MyLns2::shuffled_agents)
            .def_readwrite("old_coll_pair_num", &MyLns2::old_coll_pair_num)
            .def_readwrite("makespan", &MyLns2::makespan)
            .def_readwrite("iter_times", &MyLns2::iter_times)
            .def_readwrite("runtime", &MyLns2::runtime)
            .def_readwrite("add_sipps_path", &MyLns2::add_sipps_path)
            .def_readwrite("add_neighbor", &MyLns2::add_neighbor)
            .def_readwrite("replan_sipps_path", &MyLns2::replan_sipps_path)
            .def_readwrite("replan_neighbor", &MyLns2::replan_neighbor)
            .def_readwrite("neighbor", &MyLns2::neighbor);
}
