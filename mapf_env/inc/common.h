#pragma once
#include <tuple>
#include <list>
#include <vector>
#include <set>
#include <map>
#include <ctime>
#include <fstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <chrono>
#include <utility>
#include <boost/heap/pairing_heap.hpp>
#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <Eigen/Dense>

using boost::heap::pairing_heap;
using boost::heap::compare;
using boost::unordered_map;
using boost::unordered_set;
using std::vector;
using std::list;
using std::set;
using std::map;
using std::get;
using std::tuple;
using std::deque;
using std::make_tuple;
using std::pair;
using std::make_pair;
using std::tie;
using std::min;
using std::max;
using std::shared_ptr;
using std::make_shared;
using std::clock;
using std::cout;
using std::endl;
using std::ofstream;
using std::cerr;
using std::string;
using namespace std::chrono;
using MatrixXi = Eigen::MatrixXi;
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;