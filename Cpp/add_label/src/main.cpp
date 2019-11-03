/* 
    using nearest point to add missing labels
*/


#include <iostream>

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

struct MyPointType{
    PCL_ADD_POINT4D;
    int r;
    int g;
    int b;
    int label_orig;
    int label_pred;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (MyPointType,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (int, r, r)
                                   (int, g, g)
                                   (int, b, b)
                                   (int, label_orig, label_orig)
                                   (int, label_pred, label_pred)
)

void myReadPly(const std::string &file_name, pcl::PointCloud<MyPointType> &cloud) {
    int nSkipLine = 11; 

    std::ifstream infile(file_name.c_str());
    std::string line;

    for (int i = 0; i < nSkipLine; i++) {
        std::getline(infile, line);
    }

    float x, y, z; int r, g, b, label;  
    while (std::getline(infile, line)){
        std::istringstream iss(line);
        iss >> x >> y >> z >> r >> g >> b >> label;
        MyPointType point;
        point.x = x; point.y = y; point.z = z; point.r = r; point.g = g; point.b = b;
        point.label_orig = label; point.label_pred = -100;
        cloud.push_back(point);
    }
}

void add_miss_label(std::string orig_file, std::string pred_file, std::string save_file, int k_KNN){
    
    // int K = 1;
    std::vector<int> pointSearchIndex(k_KNN);
    std::vector<float> pointSearchDist(k_KNN);
    int *pVote = NULL;

    pcl::PointCloud<MyPointType> cloud_orig, cloud_pred;
    
    myReadPly(orig_file, cloud_orig);
    myReadPly(pred_file, cloud_pred);

    pcl::KdTreeFLANN<MyPointType> predKdTree;
    pcl::PointCloud<MyPointType>::Ptr ptrCloud(cloud_pred.makeShared());
    predKdTree.setInputCloud(ptrCloud);

    for (int i = 0; i < cloud_orig.size(); i++) {
        if (predKdTree.nearestKSearch(cloud_orig[i], k_KNN, pointSearchIndex, pointSearchDist) < k_KNN) {
            std::cerr << "Found Less Search" << std::endl;
        }

        // hard voting
        int N_labels = 20; // [0, 19]
        pVote = new int[N_labels]{0}; 
        for (int k = 1; k < k_KNN; k++) {
            pVote[cloud_pred[pointSearchIndex[k]].label_orig]++;
        }
        int maxValue = 0, maxIndex = 0;
        for (int j = 0; j < N_labels; j++) {
            if (pVote[j] > maxValue) {
                maxValue = pVote[j];
                maxIndex = j;
            }
        }

        // cloud_orig[i].label_pred = cloud_pred[pointSearchIndex[0]].label_orig; 
        cloud_orig[i].label_pred = maxIndex; 
    }

    std::ofstream out(save_file);
    for (int i = 0; i < cloud_orig.size(); i++) {
        out << cloud_orig[i].x << " " << cloud_orig[i].y << " " << cloud_orig[i].z << " "
            << (int) cloud_orig[i].r << " " << (int) cloud_orig[i].g << " " << (int) cloud_orig[i].b << " "
            << (int) cloud_orig[i].label_orig << " " << (int) cloud_orig[i].label_pred << std::endl;
    }
    out.close();
}

int main(int argc, char* argv[]) {
/*
Input:
    argv[0]: this program
    argv[1]: original_ply_file
    argv[2]: pred_ply_file
    argv[3]: save to file
    argv[4]: number of neighbors for missing label
*/

    if (argc != 5) {
        std::cerr << "[add_label] requires argc: 5, input argc: " << argc << std::endl;
        std::cerr << "Format: program orig_file pred_file save_file k_in_KNN" << std::endl;
        exit(-1);
    }

    std::string orig_file(argv[1]);
    std::string pred_file(argv[2]);
    std::string save_file(argv[3]);
    int K_KNN = atoi(argv[4]);

    add_miss_label(orig_file, pred_file, save_file, K_KNN);

    return 0;
}