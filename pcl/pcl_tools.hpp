#ifndef PCL_TOOLS_HPP
#define PCL_TOOLS_HPP

#include <iostream>
#include <fstream>

#include <pcl/io/ply_io.h> // For loading PLY files
#include <pcl/point_types.h> // For point cloud types

#if defined(ROS_VERSION) && ROS_VERSION == 2
#include <pcl_conversions/pcl_conversions.h>
#endif

#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/features/principal_curvatures.h>

namespace pcl_tools {

// Constants
constexpr int N_NEIGHBORS_SEARCH = 4;
constexpr float DRONE_RADIUS = 1.5f;

struct BoundingBox {
    float min_x, max_x, min_y, max_y, min_z, max_z;
    float width, height, depth;
    Eigen::Vector3f centroid;
};

// Point cloud saving
bool savePly(const std::string& filePath, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

// Point cloud loading
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPly(const std::string& filePath);

// Point cloud processing
template <typename PointT>
void extractPoints(
    const pcl::PointCloud<PointT>& ogCloud,
    pcl::PointCloud<PointT>& outputCloud,
    const pcl::PointIndices& indices,
    bool isExtractingOutliers
);

pcl::PointIndices removeNaNFromNormalCloud(pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud);

void removeInvalidPoints(const pcl::PointCloud<pcl::PointXYZRGB>& input,
                            pcl::PointCloud<pcl::PointXYZRGB>& output);

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const pcl::PointXYZRGB& viewPoint,
    int searchNeighbors = N_NEIGHBORS_SEARCH
);

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const pcl::PointXYZRGB& centroid
);

// Point cloud analysis
float findExtremeValue(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const std::string& targetField = "z",
    const bool findMax = false
);

pcl_tools::BoundingBox getBB(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const std::string& depthAxis = "z"
);

pcl::PointXYZRGB getHighestPoint(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud);

void decimatePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const int levels
);

void downSamplePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float leafSize
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractNeighborPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const pcl::PointXYZRGB& center,
    float radius
);

std::vector<pcl::PointIndices> extractClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float threshold,
    int minPoints
);

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float threshold,
    int minPoints
);

void removeNoise(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const int nNeighbors = 30,
    const float stdDev = 1.0,
    const bool isNegative = false
);

void smoothPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float searchRadius
);

void thresholdPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const std::string& filterFieldName,
    const float thresholdMin,
    const float thresholdMax
);

// Geometric computations
float computeDensity(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float radius);

int projectPoint(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    pcl::PointXYZRGB& point
);

pcl::PrincipalCurvatures computeCurvature(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointXYZRGB& point,
    float radius
);

Eigen::Vector4f computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

float computePlaneAngle(const Eigen::Vector4f& coefficients);

float pointToPlaneDistance(const pcl::PointXYZRGB& point, const Eigen::Vector4f& coefficients);

float computeStandardDeviation(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const Eigen::Vector4f& coefficients
);

// Distance calculations
float computePointsDist2D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

float computePointsDist3D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

std::vector<float> computeDistToCenters(
    const pcl::PointXYZRGB& landingPoint,
    const pcl::PointXYZRGB& dfCenterPoint,
    const pcl::PointXYZRGB& pcCenterPoint
);

// Visualization utilities
void printPoint(const pcl::PointXYZRGB& point);

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
    const pcl::RGB& color
);

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr coloredCloud,
    const pcl::PointIndices& inliers,
    const pcl::RGB& color
);

// Data export
void saveToCSV(
    const std::string& filename,
    const pcl::PrincipalCurvatures& curvatures,
    float density,
    float slope,
    float stdDev,
    const std::vector<float>& centerDists
);

bool checkInboundPoints(const pcl::PointXYZRGB min_pt, const pcl::PointXYZRGB max_pt, float& x, float& y);

void view(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds);

} // namespace pcl_tools

#endif // PCL_TOOLS_HPP