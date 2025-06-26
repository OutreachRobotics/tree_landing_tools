#ifndef PCL_TOOLS_HPP
#define PCL_TOOLS_HPP

#include <iostream>
#include <fstream>
#include <unordered_set>

#if defined(ROS_VERSION) && ROS_VERSION == 2
#include <pcl_conversions/pcl_conversions.h>
#endif

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/local_maximum.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace pcl_tools {

// Constants
constexpr int N_NEIGHBORS_SEARCH = 4;
constexpr float DRONE_RADIUS = 1.5f;

struct BoundingBox {
    float min_x, max_x, min_y, max_y, min_z, max_z;
    float width, height, depth;
    float volume;
    Eigen::Vector3f centroid;

    BoundingBox() {
        const float nan_val = std::numeric_limits<float>::quiet_NaN();
        min_x = max_x = min_y = max_y = min_z = max_z = nan_val;
        width = height = depth = nan_val;
        volume = nan_val;
        centroid.fill(nan_val);
    }

    BoundingBox(const Eigen::Vector3f& min_pt, const Eigen::Vector3f& max_pt):
        min_x(min_pt.x()),
        max_x(max_pt.x()),
        min_y(min_pt.y()),
        max_y(max_pt.y()),
        min_z(min_pt.z()),
        max_z(max_pt.z()),

        width(std::abs(max_pt.x() - min_pt.x())),
        height(std::abs(max_pt.y() - min_pt.y())),
        depth(std::abs(max_pt.z() - min_pt.z())),
        centroid((min_pt + max_pt) / 2.0f),
        volume(width * height * depth)
    {}
};

// Point cloud loading
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPly(const std::string& filePath);

// Point cloud saving
bool savePly(const std::string& filePath, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

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
    const pcl::PointXYZRGB& centroid,
    const int _nNeighborsSearch
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
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    const pcl::PointXYZRGB& center,
    const float radius
);

std::vector<pcl::PointIndices> extractClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float threshold,
    int minPoints
);

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud,
    float threshold,
    int minPoints=10
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

pcl::PointIndices findBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const int searchNeighbors
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const int searchNeighbors
);

pcl::PointIndices findRadiusBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointIndices boundaryIdx,
    const float searchRadius
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractRadiusBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointIndices boundaryIdx,
    const float searchRadius
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

pcl::PointIndices findLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const float radius,
    const bool isMin=false
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const float radius,
    const bool isMin=false
);

// Distance calculations
float computePointsDist2D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

float computePointsDist3D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

std::vector<std::pair<float, float>> computeDistToPointsOfInterest(
    const pcl::PointXYZRGB& landingPoint,
    const std::vector<pcl::PointXYZRGB>& pointsOfInterest
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

// Data viz
bool checkInboundPoints(const pcl::PointXYZRGB min_pt, const pcl::PointXYZRGB max_pt, float& x, float& y);

void view(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds);

} // namespace pcl_tools

#endif // PCL_TOOLS_HPP