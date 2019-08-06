#include "random.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/random_simplify_point_set.h>
#include <CGAL/property_map.h>
#include <CGAL/IO/read_xyz_points.h>

#include <vector>
#include <fstream>
#include <iostream>

#include <cmath> 

#include <boost/tuple/tuple.hpp>

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef boost::tuple<Point, float, float, float, float> MYPoint;

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

extern "C" void foo(vector<int>* v, const char* FILE_NAME){
    string line;
    ifstream myfile(FILE_NAME);
    while (getline(myfile, line)) v->push_back(1);
}

extern "C" {
    vector<int>* new_vector(){
        return new vector<int>;
    }
    void delete_vector(vector<int>* v){
        cout << "destructor called in C++ for " << v << endl;
        delete v;
    }
    int vector_size(vector<int>* v){
        return v->size();
    }
    int vector_get(vector<int>* v, int i){
        return v->at(i);
    }
    void vector_push_back(vector<int>* v, int i){
        v->push_back(i);
    }
}