#ifndef PCL_TOOLS_HPP
#define PCL_TOOLS_HPP

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <unordered_set>

#if defined(ROS_VERSION) && ROS_VERSION == 2
#include <pcl_conversions/pcl_conversions.h>
#endif

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
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
#include <pcl/segmentation/region_growing.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace pcl_tools {

// Constants
constexpr int N_NEIGHBORS_SEARCH = 30;
constexpr int HIGH_VIEW = 99999;

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

struct DepthMapData {
    cv::Mat depthMap;
    std::map<std::pair<int, int>, int> grid; // The grid-to-index map
    int min_x;
    int min_y;
    float leafSize;
};

struct DistsOfInterest {
    float distTop;

    float distTreeCenter2D;
    float distTreeCenter3D;
    float ratioTreeCenter2D;
    float ratioTreeCenter3D;

    float distTreeHighestPoint2D;
    float distTreeHighestPoint3D;
    float ratioTreeHighestPoint2D;
    float ratioTreeHighestPoint3D;
};

struct Features {
    pcl::PrincipalCurvatures curvatures;
    pcl_tools::BoundingBox treeBB;
    float density;
    float slope;
    float stdDev;
    DistsOfInterest distsOfInterest;
    pcl::ModelCoefficients::Ptr plane;
};

// Point cloud loading
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPly(const std::string& filePath);

// Point cloud saving
bool savePly(const std::string& filePath, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);

// bool saveDepthMapAsTiff(const cv::Mat& _depth_map, const std::string& _filename);

// Point cloud processing
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr extractPoints(
    typename pcl::PointCloud<PointT>::ConstPtr _cloud,
    const pcl::PointIndices& _indices,
    bool _isExtractingOutliers=false)
{
    // Create the new cloud that will be returned
    typename pcl::PointCloud<PointT>::Ptr outputCloud(new pcl::PointCloud<PointT>);

    // Check for invalid input to prevent errors
    if (!_cloud || _cloud->empty() || _indices.indices.empty()) {
        return outputCloud;
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(_cloud);
    extract.setIndices(std::make_shared<pcl::PointIndices>(_indices));
    extract.setNegative(_isExtractingOutliers);
    
    // Filter into the new cloud (dereference the pointer)
    extract.filter(*outputCloud);

    return outputCloud;
};

pcl::PointIndices removeNaNFromNormalCloud(pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud);

void removeInvalidPoints(
    const pcl::PointCloud<pcl::PointXYZRGB>& input,
    pcl::PointCloud<pcl::PointXYZRGB>& output
);

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const int _searchNeighbors
);

pcl::PointCloud<pcl::Normal>::Ptr computeNormalsRad(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _searchRadius
);

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const int searchNeighbors = N_NEIGHBORS_SEARCH
);

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const int searchNeighbors,
    const pcl::PointXYZRGB& viewPoint
);

pcl::PointIndices computeNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const float searchRadius
);

pcl::PointIndices computeNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud,
    const float searchRadius,
    const pcl::PointXYZRGB& viewPoint
);

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const int _nNeighborsSearch
);

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const int _nNeighborsSearch,
    const pcl::PointXYZRGB& _centroid
);

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const float _radiusSearch
);

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const float _radiusSearch,
    const pcl::PointXYZRGB& _centroid
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

pcl::PointIndices extractNeighborPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    const pcl::PointXYZRGB& center,
    const float radius
);

pcl::PointIndices extractNeighborCirclePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    const pcl::PointXYZRGB& center,
    const float radius
);

std::vector<pcl::PointIndices> computeClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    float threshold,
    int minPoints
);

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& pointCloud,
    const float threshold,
    const double _size_ratio=1.0,
    const int minPoints=10
);

pcl::PointCloud <pcl::PointXYZRGB>::Ptr computeSegmentation(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const int _searchNeighbors = 20,
    const float _threshNormalsAngle = 20.0,
    const float _threshCurve = 0.03
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr computeWatershed(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const float _leafSize = 0.1,
    const float _radius = 1.0,
    const float _smooth_factor = 2.0,
    const int _medianKernelSize = 5,
    const int _tophat_kernel = 9,
    const float _tophat_amplification = 10.0,
    const float _pacman_solidity = 0.6,
    const bool _shouldView = false,
    std::optional<pcl::PointXYZRGB> _point = std::nullopt
);

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extractClusters(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const std::vector<pcl::PointIndices>& _clusters
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractClosestCluster(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const std::vector<pcl::PointIndices>& _clusters,
    const pcl::PointXYZRGB& _point
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateGridCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float& _radius,
    const float& _radius_factor = 0.2,
    const float _max_ratio_from_center = 0.5
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

void extractBoundary(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr& normalsCloud,
    const int searchNeighbors
);

pcl::PointIndices findRadiusBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointIndices boundaryIdx,
    const float searchRadius
);

void extractRadiusBoundary(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const pcl::PointIndices _boundaryIdx,
    const float _searchRadius,
    const bool _should_invert = false
);

pcl::PointIndices findSurface(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const float leafSize
);

void extractSurface(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const float leafSize
);

// Geometric computations
float computeDensity(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, const float radius);

float computeSurfaceDensity(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const float radius
);

float projectPoint(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& _cloud,
    pcl::PointXYZRGB& _point,
    const int _numNeighbors = 12
);

pcl::PrincipalCurvatures computeCurvature(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::PointXYZRGB& point,
    float radius
);

// Eigen::Vector4f computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
pcl::ModelCoefficients::Ptr computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

float computePlaneAngle(const pcl::ModelCoefficients::Ptr coefficients);

float pointToPlaneDistance(const pcl::PointXYZRGB& point, const pcl::ModelCoefficients::Ptr coefficients);

float computeStandardDeviation(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const pcl::ModelCoefficients::Ptr coefficients
);

pcl::PointIndices findLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
    const float radius,
    const bool isMin=false
);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
    const float radius,
    const bool isMin=false
);

// Distance calculations
float computePointsDist2D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

float computePointsDist3D(const pcl::PointXYZRGB& point1, const pcl::PointXYZRGB& point2);

DistsOfInterest computeDistToPointsOfInterest(
    const pcl::PointXYZRGB& _landingPoint,
    const std::vector<pcl::PointXYZRGB>& _pointsOfInterest,
    const pcl_tools::BoundingBox& _treeBB
);

Features computeFeatures(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _landingSurfaceCloud,
    const float& _lz_factor,
    const float& _radius
);

Features computeLandingPointFeatures(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud
);

Features computeLandingZoneFeatures(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _landingSurfaceCloud,
    const float& _lz_factor,
    const float& _radius,
    const Features& _tree_features
);

std::vector<pcl_tools::Features> computeFeaturesList(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _gridCloud,
    const float& _landing_zone_factor,
    const float& _radius,
    const float& _min_lz_points
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
bool checkInboundPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _ogCloud, const std::vector<float>& landing);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud);

void view(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> clouds,
    const pcl::ModelCoefficients::Ptr _plane = nullptr,
    const pcl::PointXYZRGB* _sphere = nullptr
);

} // namespace pcl_tools

#endif // PCL_TOOLS_HPP