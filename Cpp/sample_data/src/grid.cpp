#include "grid.hpp"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/IO/read_xyz_points.h>

#include <vector>
#include <fstream>
#include <iostream>

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;

void grid(int argc, char*argv[]) {

    if (argc != 7) {
        std::cerr << "argc: " << argc << " input argc = " << argc << std::endl;
    }

    std::string sampler(argv[1]);
    int start = atoi(argv[2]);
    int end = atoi(argv[3]);
    int step = atoi(argv[4]);
    std::string dstDir(argv[5]);
    std::string dstFile(argv[6]);

    /*
    std::cout << "sampler: " << sampler
        << ", start: " << start << ", end: " << end << ", step: " << step 
        << ", dstDir: " << dstDir << ", dstFile: " << dstFile << std::endl;
    */

    std::vector<Point> points;

    std::string srcFilename = "../tmp/" + dstFile;

    // read points 
    // std::cout << "Reading file: " << srcFilename << std::endl;
    std::FILE *pFile = fopen(srcFilename.c_str(), "rb");
    fseek(pFile, 0, SEEK_END);    // file pointer goes to the end of the file
    long fileSize = ftell(pFile); // file size
    rewind(pFile);                // rewind file pointer to the beginning
    float *rawData = new float[fileSize];
    fread(rawData, sizeof(float), fileSize/sizeof(float), pFile);
    long number_of_points = fileSize / 3 / sizeof(float); // x, y, z

    for (int i = 0; i < number_of_points; i++) {
        points.push_back(Point(rawData[3*i], rawData[3*i+1], rawData[3*i+2]));
    }

    std::cout << "Point number of points: " << points.size() << std::endl;

    int n_sample = 100;
    double max_cell_size = 0.1;
    std::vector<double> cell_size_vec;
    std::vector<int> cell_number_points_vec;
    for (int i = 0; i < n_sample; i++) {
        cell_size_vec.push_back(max_cell_size / n_sample * (i+1));
        cell_number_points_vec.push_back(0);
    }

    for (int i = 0; i < cell_size_vec.size(); i++) {
        std::vector<Point> points_copy = points;
        points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), cell_size_vec[i]),
                points_copy.end());
        cell_number_points_vec[i] = points_copy.size();
    }

    for (int keep_ratio = start; keep_ratio <= end; keep_ratio += step) {
        int n_keep_points = (int)(keep_ratio / 100.0 * number_of_points);

        for (int i = 0; i < cell_number_points_vec.size(); i++) {
            if (n_keep_points <= cell_number_points_vec[i]) {
                std::vector<Point> points_copy = points;
                points_copy.erase(CGAL::grid_simplify_point_set(points_copy.begin(), points_copy.end(), cell_size_vec[i]),
                    points_copy.end());
                // save to file
                std::string dstFileSave = srcFilename + "." + std::to_string(keep_ratio);
                // std::cout << dstFileSave << " " << points_copy.size() << std::endl;
                std::ofstream out(dstFileSave, std::ios_base::binary);
                for (int i = 0; i < points_copy.size(); i++) {
                    float x = points_copy[i].x();
                    float y = points_copy[i].y();
                    float z = points_copy[i].z();
                    out.write((char *)&x, sizeof(float));
                    out.write((char *)&y, sizeof(float));
                    out.write((char *)&z, sizeof(float));
                }
                out.close();
                break;
            }
        }
    }
}