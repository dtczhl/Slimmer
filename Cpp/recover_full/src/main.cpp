/* 
    creating labels using partial original labels
*/

#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <fstream>

#define PCL_NO_PRECOMPILE
#include <pcl/pcl_macros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/random_sample.h>


struct MyPointType{
    PCL_ADD_POINT4D;
    int r;
    int g;
    int b;
    int label_orig;
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
)

struct ConfusionMatrix {
    int TP, TN, FP, FN;

    ConfusionMatrix() {
        this->TP = 0;
        this->TN = 0;
        this->FP = 0;
        this->FN = 0;
    }
};

// list files in a directory
void list_files_in_dir(std::string folder, std::vector<std::string> &filenames, std::string suffix = "") {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir(folder.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.length() <  suffix.length() || filename.substr(filename.length() - suffix.length()).compare(suffix) != 0) {
                continue;
            }
            filenames.push_back(filename);
        }
    } else {
        std::cerr << "Cannot open " << folder << std::endl;
        exit(-1);
    }
}

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
        if (label < 0 || label >= 20) continue;
        MyPointType point;
        point.x = x; point.y = y; point.z = z; point.r = r; point.g = g; point.b = b;
        point.label_orig = label;
        cloud.push_back(point);
    }
}

void recover_full_from_partial(std::string ply_file, int K, std::string save_file) {

    std::ofstream out_file(save_file.c_str());

    out_file << "keep_ratio(%),class_ID,true_positive(TP),true_negative(TN),false_positive(FP),false_negative(FN)" << std::endl;
    
    pcl::PointCloud<MyPointType> cloud_full, cloud_partial;
    myReadPly(ply_file, cloud_full);

    pcl::RandomSample<MyPointType> random_sample;
    pcl::PointCloud<MyPointType>::Ptr ptrCloudFull(cloud_full.makeShared());
    random_sample.setInputCloud(cloud_full.makeShared());

    std::vector<int> pointSearchIndex(K);
    std::vector<float> pointSearchDist(K);
    int *pVote = NULL;
    pcl::KdTreeFLANN<MyPointType> predKdTree;

    int N_labels = 20; // [0, 19]

    for (int i_ratio = 1; i_ratio <= 100; i_ratio++) {
        random_sample.setSample(cloud_full.size() * i_ratio / 100.0);
        random_sample.filter(cloud_partial);

        predKdTree.setInputCloud(cloud_partial.makeShared());
        std::vector<ConfusionMatrix> confusion_matrices(N_labels);

        for (int i_point = 0; i_point < cloud_full.size(); i_point++) {

            bool bPointOriginal = false;

            if (cloud_full[i_point].label_orig < 0 || 
                    cloud_full[i_point].label_orig >= N_labels) continue;

            if (predKdTree.nearestKSearch(cloud_full[i_point], K, pointSearchIndex, pointSearchDist) < K) {
                std::cerr << "Found Less Search" << std::endl;
            }

            if (pointSearchDist[0] == 0) { // points in cloud_partial
                bPointOriginal = true;
            }

            int maxValue = 0, maxIndex = 0;
            if (!bPointOriginal) {
            // hard voting
                pVote = new int[N_labels]{0}; 
                for (int k = 0; k < K; k++) {
                    if (cloud_partial[pointSearchIndex[k]].label_orig < 0 || 
                        cloud_partial[pointSearchIndex[k]].label_orig >= N_labels) {
                            continue;
                        };
                    pVote[cloud_partial[pointSearchIndex[k]].label_orig]++;
                }

                for (int j = 0; j < N_labels; j++) {
                    if (pVote[j] > maxValue) {
                        maxValue = pVote[j];
                        maxIndex = j;
                    }
                }
            } else {
                maxIndex = cloud_partial[pointSearchIndex[0]].label_orig;  // points in cloud_partial
            }

            if (cloud_full[i_point].label_orig == maxIndex) {
                // true positive
                confusion_matrices[cloud_full[i_point].label_orig].TP++;
            } else {
                // false negative
                confusion_matrices[cloud_full[i_point].label_orig].FN++;
                // false positive
                confusion_matrices[maxIndex].FP++;
            }
        }

        for (int i_label = 0; i_label < N_labels; i_label++) {
            out_file << i_ratio << "," << i_label << "," 
                << confusion_matrices[i_label].TP << "," << confusion_matrices[i_label].TN << "," 
                << confusion_matrices[i_label].FP << "," << confusion_matrices[i_label].FN << std::endl;
        } 
    }
    out_file.flush();
    out_file.close(); 
}

int main(int argc, char* argv[]) {
/*
Input:
    argv[0]: this program
    argv[1]: ply_file
    argv[2]: K nearest neighbor
    argv[3]: save_file
*/

    if (argc != 4) {
        std::cerr << "[recover_full] requires argc: 4, input argc: " << argc << std::endl;
        std::cerr << "Format: program ply_file K save_file" << std::endl;
        exit(-1);
    }

    std::string ply_file(argv[1]);
    int K = atoi(argv[2]);
    std::string save_file(argv[3]);

    recover_full_from_partial(ply_file, K, save_file);

    return 0;
}