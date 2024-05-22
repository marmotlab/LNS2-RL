#include "./inc/mapf_env.h"
#include "./pybind11/include/pybind11/pybind11.h"
#include "dynamic_state.cpp"
#include "mapf_env.cpp"
#include "static_state.cpp"
// to convert C++ STL containers to python list
#include "./pybind11/include/pybind11/stl.h"
#include <pybind11/eigen.h>
#include <Eigen/Dense>

namespace py=pybind11;
using namespace std;

PYBIND11_MODULE(my_env, m) {
    py::class_<MapfEnv>(m, "MapfEnv")
            .def(py::init<int, int, int, int, int, double, int ,int, int,pair<int, int>,pair<double, double>,vector<vector<pair<int, int>>>,
                 MatrixXi,std::map<pair<int, int>, vector<int>>,MatrixXi,vector<pair<int, int>>, vector<pair<int, int>>>())
            .def("observe", &MapfEnv::observe)
            .def("predict_next", &MapfEnv::predict_next)
            .def("update_ulti", &MapfEnv::update_ulti)
            .def("joint_step", &MapfEnv::joint_step)
            .def("local_reset", &MapfEnv::local_reset)
            .def("replan_part1", &MapfEnv::replan_part1)
            .def("replan_part2", &MapfEnv::replan_part2)
            .def("replan_part3", &MapfEnv::replan_part3)
            .def("next_valid_actions", &MapfEnv::next_valid_actions)
            .def_readwrite("rupt", &MapfEnv::rupt)
            .def_readwrite("new_collision_pairs", &MapfEnv::new_collision_pairs)
            .def_readwrite("local_path", &MapfEnv::local_path)
            .def_readwrite("old_path", &MapfEnv::old_path)
            .def_readwrite("replan_ag", &MapfEnv::replan_ag)
            .def_readwrite("all_obs", &MapfEnv::all_obs)
            .def_readwrite("all_vector", &MapfEnv::all_vector)
            .def_readwrite("timestep", &MapfEnv::timestep)
            .def_readwrite("agents_poss", &MapfEnv::agents_poss)
            .def_readwrite("valid_actions", &MapfEnv::valid_actions);
}


