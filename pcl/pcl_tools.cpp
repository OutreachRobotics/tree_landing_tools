#include "pcl_tools.hpp"

namespace pcl_tools {

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPly(const std::string& _filePath)
{
    // Create a point cloud object for XYZRGB points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Load the .ply file
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(_filePath, *cloud) == -1) {
        std::cout << "Error: Could not load .ply file: " << _filePath << std::endl;
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr();
    }

    // Print the number of points in the cloud
    std::cout << "Loaded " << cloud->width * cloud->height << " points from " << _filePath << std::endl;

    // Print the first 5 points with their RGB values (optional)
    for (size_t i = 0; i < 5 && i < cloud->points.size(); ++i) {
        const auto& point = cloud->points[i];
        std::cout << "Point " << i << ": "
                  << "X: " << point.x << ", "
                  << "Y: " << point.y << ", "
                  << "Z: " << point.z << ", "
                  << "R: " << static_cast<int>(point.r) << ", "
                  << "G: " << static_cast<int>(point.g) << ", "
                  << "B: " << static_cast<int>(point.b) << std::endl;
    }

    return cloud;
}

bool savePly(const std::string& _filePath, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud)
{
    if (!_cloud || _cloud->empty()) {
        std::cout << "Error: Point cloud is empty or invalid." << std::endl;
        return false;
    }

    std::cout << "Saving " << _cloud->points.size() << " points to " << _filePath << std::endl;

    // Save the point cloud to a .ply file in ASCII format
    if (pcl::io::savePLYFileASCII(_filePath, *_cloud) == -1) {
        std::cout << "Error: Could not save .ply file: " << _filePath << std::endl;
        return false;
    }

    std::cout << "Successfully saved point cloud to " << _filePath << std::endl;
    return true;
}

bool saveDepthMapAsTiff(const cv::Mat& _depth_map, const std::string& _filePath)
{
    // Ensure the matrix is not empty before saving
    if (_depth_map.empty()) {
        std::cout << "Error: Can't save empty depth map." << std::endl;
        return false;
    }

    cv::imwrite(_filePath, _depth_map);
    std::cout << "Successfully saved point cloud to " << _filePath << std::endl;
    return true;
}

pcl::PointIndices removeNaNFromNormalCloud(pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud)
{
    pcl::PointCloud<pcl::PointNormal> tempCloud;
    tempCloud.points.reserve(_normalsCloud->points.size());

    // Create a PointIndices object to store the removed indices
    pcl::PointIndices removedIndices;
    removedIndices.indices.reserve(_normalsCloud->points.size());

    int i = 0;
    for (const auto& normal : _normalsCloud->points) {
        if (!std::isnan(normal.normal_x) && !std::isnan(normal.normal_y) && !std::isnan(normal.normal_z))
        {
            tempCloud.emplace_back(normal);
        }
        else
        {
            // Add index of the point with NaN value to removedIndices
            removedIndices.indices.emplace_back(i);
        }
        ++i;
    }

    pcl::copyPointCloud(tempCloud, *_normalsCloud);
    return removedIndices;
}

void removeInvalidPoints(const pcl::PointCloud<pcl::PointXYZRGB>& _input,
                            pcl::PointCloud<pcl::PointXYZRGB>& _output)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(_input, *temp);

    pcl::PointCloud<pcl::PointXYZRGB> output;

    output.reserve(temp->size());
    for (size_t i = 0; i < temp->size(); ++i) {
        const auto& p = temp->points[i];
        if (pcl::isFinite(p)) {  // Checks both NaN AND Inf
            output.emplace_back(p);
        }
    }
    output.width = output.size();
    output.height = 1;
    output.is_dense = false;
    
    pcl::copyPointCloud(output, _output);
}

pcl::PointCloud<pcl::Normal>::Ptr computeNormals(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const int _searchNeighbors)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_cloud);

    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setSearchMethod(kdtree);
    ne.setInputCloud(_cloud);
    ne.setViewPoint(0, 0, HIGH_VIEW);
    ne.setKSearch(_searchNeighbors);
    ne.compute(*normals);

    return normals;
}

pcl::PointCloud<pcl::Normal>::Ptr computeNormalsRad(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _searchRadius)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_cloud);

    pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setSearchMethod(kdtree);
    ne.setInputCloud(_cloud);
    ne.setViewPoint(0, 0, HIGH_VIEW);
    ne.setRadiusSearch(_searchRadius);
    ne.compute(*normals);

    return normals;
}

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const int _searchNeighbors)
{
    pcl::PointXYZRGB viewPoint;
    viewPoint.z = HIGH_VIEW;
    return computeNormalsPC(
        _pointCloud,
        _normalsCloud,
        _searchNeighbors,
        viewPoint
    );
}

pcl::PointIndices computeNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const int _searchNeighbors,
    const pcl::PointXYZRGB& _viewPoint)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(_pointCloud);

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
    ne.setInputCloud(_pointCloud);
    ne.setViewPoint(_viewPoint.x, _viewPoint.y, _viewPoint.z);
    ne.setSearchMethod(kdTree);
    ne.setKSearch(_searchNeighbors);
    ne.compute(*_normalsCloud);

    pcl::PointIndices removedIdx = removeNaNFromNormalCloud(_normalsCloud);
    _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointIndices computeNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const float _searchRadius)
{
    pcl::PointXYZRGB viewPoint;
    viewPoint.z = HIGH_VIEW;
    return computeNormalsRadPC(
        _pointCloud,
        _normalsCloud,
        _searchRadius,
        viewPoint
    );
}

pcl::PointIndices computeNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const float _searchRadius,
    const pcl::PointXYZRGB& _viewPoint)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(_pointCloud);

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::PointNormal> ne;
    ne.setInputCloud(_pointCloud);
    ne.setViewPoint(_viewPoint.x, _viewPoint.y, _viewPoint.z);
    ne.setSearchMethod(kdTree);
    ne.setRadiusSearch(_searchRadius);
    ne.compute(*_normalsCloud);

    pcl::PointIndices removedIdx = removeNaNFromNormalCloud(_normalsCloud);
    _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const int _nNeighborsSearch)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsPC(_pointCloud, normalsCloud, _nNeighborsSearch);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const int _nNeighborsSearch,
    const pcl::PointXYZRGB& _centroid)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsPC(_pointCloud, normalsCloud, _nNeighborsSearch, _centroid);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const float _radiusSearch)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsRadPC(_pointCloud, normalsCloud, _radiusSearch);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const float _radiusSearch,
    const pcl::PointXYZRGB& _centroid)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsRadPC(_pointCloud, normalsCloud, _radiusSearch, _centroid);
    return normalsCloud;
}

float findExtremeValue(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const std::string& _targetField,
    const bool _findMax)
{
    if (!_pointCloud || _pointCloud->empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    float extremeValue = _findMax ? std::numeric_limits<float>::min() 
                        : std::numeric_limits<float>::max();
    bool foundValidValue = false;

    // Iterate through all points
    for (const auto& point : *_pointCloud) {
        const float currentValue = 
            (_targetField == "x") ? point.x :
            (_targetField == "y") ? point.y :
            point.z;  // Default case

        if (!std::isnan(currentValue)) {
            if (_findMax) {
                if (currentValue > extremeValue) {
                    extremeValue = currentValue;
                    foundValidValue = true;
                }
            } else {
                if (currentValue < extremeValue) {
                    extremeValue = currentValue;
                    foundValidValue = true;
                }
            }
        }
    }

    return foundValidValue ? extremeValue : std::numeric_limits<float>::quiet_NaN();
}

pcl_tools::BoundingBox getBB(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const std::string& _depthAxis)
{
    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D(*_pointCloud, minPt, maxPt);
  
    // std::cout << "Min X: " << minPt.x << ", Min Y: " << minPt.y << ", Min Z: " << minPt.z << std::endl;
    // std::cout << "Max X: " << maxPt.x << ", Max Y: " << maxPt.y << ", Max Z: " << maxPt.z << std::endl;
  
    pcl_tools::BoundingBox boundingBox;
    boundingBox.min_x = minPt.x;
    boundingBox.max_x = maxPt.x;
    boundingBox.min_y = minPt.y;
    boundingBox.max_y = maxPt.y;
    boundingBox.min_z = minPt.z;
    boundingBox.max_z = maxPt.z;
    boundingBox.centroid = Eigen::Vector3f((minPt.x + maxPt.x) / 2.0f,
                                            (minPt.y + maxPt.y) / 2.0f,
                                            (minPt.z + maxPt.z) / 2.0f);

    boundingBox.width = maxPt.x - minPt.x;
    boundingBox.height = maxPt.y - minPt.y;
    boundingBox.depth = maxPt.z - minPt.z;

    if (_depthAxis == "x") {
        boundingBox.width = maxPt.y - minPt.y;
        boundingBox.height = maxPt.z - minPt.z;
        boundingBox.depth = maxPt.x - minPt.x;
    }

    boundingBox.volume = boundingBox.width * boundingBox.height * boundingBox.depth;

    return boundingBox;
}

double distanceToOBBCenter2D(
    const pcl::PointXYZRGB& _point,
    const OrientedBoundingBox& _obb)
{
    Eigen::Vector2f difference = _point.getVector3fMap().head<2>() - _obb.centroid.head<2>();
    return difference.norm();
}

double distanceToOBB2D(
    const pcl::PointXYZRGB& _point,
    const OrientedBoundingBox& _obb)
{
    // Step 1: Transform the point into the OBB's local coordinate system.
    // This step remains the same as the 3D version.
    Eigen::Vector3f point_translated = _point.getVector3fMap() - _obb.centroid;
    Eigen::Vector3f point_local = _obb.rotation.conjugate() * point_translated;

    // Step 2: Calculate the distance from the local point to the AABB,
    // but only for the X and Y axes.
    Eigen::Vector3f half_extents(_obb.width / 2.0f, _obb.height / 2.0f, _obb.depth / 2.0f);

    // d is the vector from the point to the box boundary.
    // Its components will be negative if the point is inside on that axis.
    Eigen::Vector2f d = point_local.head<2>().cwiseAbs() - half_extents.head<2>();

    // Step 3: Calculate the final signed distance.
    // This combines the external distance (from positive components of d)
    // and the internal distance (from the largest negative component of d).
    double outside_distance = d.cwiseMax(0.0f).norm();
    double inside_distance = std::min(0.0f, d.maxCoeff());

    return outside_distance + inside_distance;
}

OrientedBoundingBox getOBB(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud)
{
    // 1. Compute the moment of inertia and extraction class
    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor;
    feature_extractor.setInputCloud(_pointCloud);
    feature_extractor.compute();

    // 2. Define variables to store OBB properties
    // *FIX*: The position variable must match the point type of the cloud
    pcl::PointXYZRGB obb_position; 
    Eigen::Matrix3f obb_rotation_matrix;
    pcl::PointXYZRGB min_point_OBB;
    pcl::PointXYZRGB max_point_OBB;

    // 3. Get the OBB
    // This call will now work correctly
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, obb_position, obb_rotation_matrix);
    
    // 4. Create a quaternion from the rotation matrix
    Eigen::Quaternionf obb_quaternion(obb_rotation_matrix);

    // 5. Construct the return object
    OrientedBoundingBox obb;
    obb.centroid = Eigen::Vector3f(obb_position.x, obb_position.y, obb_position.z);
    obb.rotation = obb_quaternion.normalized();
    
    // 6. Calculate dimensions
    obb.width = max_point_OBB.x - min_point_OBB.x;
    obb.height = max_point_OBB.y - min_point_OBB.y;
    obb.depth = max_point_OBB.z - min_point_OBB.z;
    
    return obb;
}

OrientedBoundingBox getZAlignedOBB(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _hullCloud
){
    // 2. Compute moment of inertia on the 2D HULL, not the original cloud
    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor;
    feature_extractor.setInputCloud(_hullCloud);
    feature_extractor.compute();

    // 3. Define variables for the hull's OBB
    pcl::PointXYZRGB obb_position_hull; 
    Eigen::Matrix3f obb_rotation_matrix_3D;
    pcl::PointXYZRGB min_point_OBB_hull;
    pcl::PointXYZRGB max_point_OBB_hull;

    // 4. Get the OBB (this will be slightly tilted, which we'll fix)
    feature_extractor.getOBB(min_point_OBB_hull, max_point_OBB_hull, obb_position_hull, obb_rotation_matrix_3D);
    
    // --- 5. FORCE Z-ALIGNMENT ---
    
    // Get the OBB's main X-axis from the 3D rotation
    Eigen::Vector3f x_axis_3D = obb_rotation_matrix_3D.col(0);

    // Calculate the 2D yaw angle by projecting the X-axis onto the XY plane
    // This gives us the pure Z-rotation and ignores any X/Y tilt.
    float yaw_z_angle = std::atan2(x_axis_3D.y(), x_axis_3D.x());

    // Create a new, clean quaternion using ONLY the Z-rotation
    Eigen::Quaternionf z_aligned_rotation(Eigen::AngleAxisf(yaw_z_angle, Eigen::Vector3f::UnitZ()));

    // --- 6. GET Z-RANGE from the original 3D cloud ---
    pcl::PointXYZRGB min_point_cloud;
    pcl::PointXYZRGB max_point_cloud;
    pcl::getMinMax3D(*_pointCloud, min_point_cloud, max_point_cloud);

    // 7. Construct the final Z-aligned OBB
    OrientedBoundingBox obb;
    
    // Centroid: X/Y from the hull's OBB, Z from the full cloud's center
    obb.centroid = Eigen::Vector3f(
        obb_position_hull.x, 
        obb_position_hull.y, 
        (min_point_cloud.z + max_point_cloud.z) / 2.0f
    );

    // Rotation: The new, clean Z-aligned rotation
    obb.rotation = z_aligned_rotation.normalized();
    
    // Dimensions: X/Y from the hull's OBB, Z from the full cloud's range
    obb.width = max_point_OBB_hull.x - min_point_OBB_hull.x;
    obb.height = max_point_OBB_hull.y - min_point_OBB_hull.y;
    obb.depth = max_point_cloud.z - min_point_cloud.z;
    
    return obb;
}

OrientedBoundingBox getZAlignedOBB(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud, const double _CONCAVE_HULL_ALPHA)
{
    // 1. Get the 2D hull (footprint)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_polygon = pcl_tools::computeConcaveHull2D(
        _pointCloud,
        _CONCAVE_HULL_ALPHA
    );

    // 2. Compute moment of inertia on the 2D HULL, not the original cloud
    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor;
    feature_extractor.setInputCloud(hull_polygon); // <-- Use hull_polygon
    feature_extractor.compute();

    // 3. Define variables for the hull's OBB
    pcl::PointXYZRGB obb_position_hull; 
    Eigen::Matrix3f obb_rotation_matrix_3D;
    pcl::PointXYZRGB min_point_OBB_hull;
    pcl::PointXYZRGB max_point_OBB_hull;

    // 4. Get the OBB (this will be slightly tilted, which we'll fix)
    feature_extractor.getOBB(min_point_OBB_hull, max_point_OBB_hull, obb_position_hull, obb_rotation_matrix_3D);
    
    // --- 5. FORCE Z-ALIGNMENT ---
    
    // Get the OBB's main X-axis from the 3D rotation
    Eigen::Vector3f x_axis_3D = obb_rotation_matrix_3D.col(0);

    // Calculate the 2D yaw angle by projecting the X-axis onto the XY plane
    // This gives us the pure Z-rotation and ignores any X/Y tilt.
    float yaw_z_angle = std::atan2(x_axis_3D.y(), x_axis_3D.x());

    // Create a new, clean quaternion using ONLY the Z-rotation
    Eigen::Quaternionf z_aligned_rotation(Eigen::AngleAxisf(yaw_z_angle, Eigen::Vector3f::UnitZ()));

    // --- 6. GET Z-RANGE from the original 3D cloud ---
    pcl::PointXYZRGB min_point_cloud;
    pcl::PointXYZRGB max_point_cloud;
    pcl::getMinMax3D(*_pointCloud, min_point_cloud, max_point_cloud);

    // 7. Construct the final Z-aligned OBB
    OrientedBoundingBox obb;
    
    // Centroid: X/Y from the hull's OBB, Z from the full cloud's center
    obb.centroid = Eigen::Vector3f(
        obb_position_hull.x, 
        obb_position_hull.y, 
        (min_point_cloud.z + max_point_cloud.z) / 2.0f
    );

    // Rotation: The new, clean Z-aligned rotation
    obb.rotation = z_aligned_rotation.normalized();
    
    // Dimensions: X/Y from the hull's OBB, Z from the full cloud's range
    obb.width = max_point_OBB_hull.x - min_point_OBB_hull.x;
    obb.height = max_point_OBB_hull.y - min_point_OBB_hull.y;
    obb.depth = max_point_cloud.z - min_point_cloud.z;
    
    return obb;
}

pcl::PointXYZRGB getHighestPoint(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud)
{
    // Initialize the highest point with the first point in the cloud
    pcl::PointXYZRGB highestPoint = _pointCloud->points[0];

    // Iterate through the cloud to find the highest point
    for (const auto& point : _pointCloud->points) {
        if (point.z > highestPoint.z) {
            highestPoint = point;
        }
    }

    return highestPoint;
}

void decimatePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const int _levels)
{
    if (_levels <= 0){return;}

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr dsCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    dsCloud->width = (_pointCloud->width  + _levels - 1) / _levels;
    dsCloud->height = (_pointCloud->height  + _levels - 1) / _levels;
    dsCloud->resize(dsCloud->width * dsCloud->height);
    
    for (uint32_t y = 0; y < _pointCloud->height; y += _levels) {
      for (uint32_t x = 0; x < _pointCloud->width; x += _levels) {
        (*dsCloud)(x/_levels, y/_levels) = (*_pointCloud)(x, y);
      }
    }
    dsCloud->is_dense = _pointCloud->is_dense;
    pcl::copyPointCloud(*dsCloud, *_pointCloud);
}

void downSamplePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const float _leafSize)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(_pointCloud);
    sor.setLeafSize(_leafSize, _leafSize, _leafSize);
    sor.filter(*_pointCloud);
}

pcl::PointIndices extractNeighborPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const pcl::PointXYZRGB& _center,
    const float _radius)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_pointCloud);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    pcl::PointIndices inliers;
    if(kdtree->radiusSearch(_center, _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
        inliers.indices = pointIdxRadiusSearch;
    }

    _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, inliers, false);

    return inliers;
}

pcl::PointIndices extractNeighborCirclePC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const pcl::PointXYZRGB& _center,
    const float _radius)
{
    if(_pointCloud->size() == 0) {
        std::cout << "extractNeighborCirclePC returning empty indices after recieving empty cloud" << std::endl;
        return pcl::PointIndices();
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr flattenPC(new pcl::PointCloud<pcl::PointXYZRGB>(*_pointCloud));

    pcl::PointXYZRGB searchCenter = _center;
    searchCenter.z = 0.0f;

    for (auto& point : *flattenPC) {
        // Set the Z coordinate to 0
        point.z = 0.0f;
    }

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(flattenPC);

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    pcl::PointIndices inliers;
    if(kdtree->radiusSearch(searchCenter, _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
        inliers.indices = pointIdxRadiusSearch;
    }

    _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, inliers, false);

    return inliers;
}

pcl::PointIndices concatenateClusters(
    const std::vector<pcl::PointIndices>& _cluster_indices) 
{
    pcl::PointIndices combined;
    std::unordered_set<int> unique_indices;  // Automatically handles duplicates
    
    // Reserve approximate space (optional optimization)
    size_t total_points = 0;
    for (const auto& cluster : _cluster_indices) {
        total_points += cluster.indices.size();
    }
    unique_indices.reserve(total_points);
    
    // Insert all indices (duplicates automatically ignored)
    for (const auto& cluster : _cluster_indices) {
        unique_indices.insert(cluster.indices.begin(), cluster.indices.end());
    }
    
    // Copy to output
    combined.indices.assign(unique_indices.begin(), unique_indices.end());
    return combined;
}

std::vector<pcl::PointIndices> computeClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    float _threshold,
    int _minPoints)
{
    std::vector<pcl::PointIndices> cluster_indices;
    
    // Create KD-tree for searching
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    tree->setInputCloud(_pointCloud);

    // Euclidean clustering
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(_threshold);
    ec.setMinClusterSize(_minPoints);
    ec.setMaxClusterSize(std::numeric_limits<int>::max());
    ec.setSearchMethod(tree);
    ec.setInputCloud(_pointCloud);
    ec.extract(cluster_indices);

    if (cluster_indices.size() > 0)
    {
        _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, concatenateClusters(cluster_indices), false);
    }

    return cluster_indices;
}

pcl::PointIndices extractBiggestSegment(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const std::vector<pcl::PointIndices> _segment_indices,
    const double _size_ratio=1.0)
{
    size_t max_cluster_size = 0;
    for (const auto& segment : _segment_indices)
    {
        if (segment.indices.size() > max_cluster_size)
        {
            max_cluster_size = segment.indices.size();
        }
    }

    const size_t size_threshold = static_cast<size_t>(max_cluster_size * _size_ratio);
    std::cout << "Size threshold: " << size_threshold << std::endl;

    pcl::PointIndices combined_inliers;
    size_t qualifying_clusters_count = 0;
    for (const auto& segment : _segment_indices)
    {
        std::cout << "Segment size: " << segment.indices.size() << std::endl;
        if (segment.indices.size() >= size_threshold)
        {
            // Add the indices of this qualifying cluster to our combined list
            combined_inliers.indices.insert(
                combined_inliers.indices.end(),
                segment.indices.begin(),
                segment.indices.end()
            );
            qualifying_clusters_count++;
        }
    }

    if(qualifying_clusters_count > 0)
    {
        _pointCloud = extractPoints<pcl::PointXYZRGB>(_pointCloud, combined_inliers, false);
        std::cout << "The point cloud has " << std::to_string(_segment_indices.size()) << " clusters." << std::endl;
        std::cout << "And " << qualifying_clusters_count << " clusters were kept." << std::endl;
    }
    else
    {
        std::cout << "No clusters found in the point cloud." << std::endl;
    } 

    return combined_inliers;
}

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _pointCloud,
    const float _threshold,
    const double _size_ratio,
    const int _minPoints)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_pointCloud));

    // Cluster extraction
    std::vector<pcl::PointIndices> cluster_indices = computeClusters(
        tempPointCloud,
        _threshold,
        _minPoints
    );

    return extractBiggestSegment(_pointCloud, cluster_indices, _size_ratio);
}

pcl::PointCloud <pcl::PointXYZRGB>::Ptr computeSegmentation(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const int _searchNeighbors,
    const float _threshNormalsAngle,
    const float _threshCurve)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals(_cloud, _searchNeighbors);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*_cloud, *cloud_xyz);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_xyz(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree_xyz->setInputCloud(cloud_xyz);

    pcl::RegionGrowing<pcl::PointXYZ,pcl::Normal> reg;
    reg.setMinClusterSize(40);
    reg.setSearchMethod(kdtree_xyz);
    reg.setNumberOfNeighbours(_searchNeighbors);
    reg.setInputCloud(cloud_xyz);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(_threshNormalsAngle * M_PI/180.0);
    reg.setCurvatureThreshold(_threshCurve);

    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters is equal to " << clusters.size() << std::endl;

    for(size_t i=0; i < clusters.size(); ++i) {
        std::cout << i << " cluster has " << clusters[i].indices.size () << " points." << std::endl;
    }

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
    return colored_cloud;
}

std::map<std::pair<int, int>, int> computeGrid(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _leafSize)
{
    // 1. Create the map to store the highest point for each grid cell
    // The key is the grid cell coordinate, the value is the index of the point
    std::map<std::pair<int, int>, int> grid;

    // 2. Iterate through the cloud and find the highest point in each cell
    for (int i = 0; i < _cloud->points.size(); ++i) {
        int grid_x = static_cast<int>(floor(_cloud->points[i].x / _leafSize));
        int grid_y = static_cast<int>(floor(_cloud->points[i].y / _leafSize));
        std::pair<int, int> cell = std::make_pair(grid_x, grid_y);

        // Check if a point is already in this cell
        if (grid.find(cell) == grid.end()) {
            // No point yet, so add this one
            grid[cell] = i;
        } else {
            // A point exists, check if the new one is higher
            int idx = grid[cell];

            if (_cloud->points[i].z > _cloud->points[idx].z) {
                // New point is higher, replace the old one
                grid[cell] = i;
            }
        }
    }

    return grid;
}

DepthMapData computeDepthMap(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _leafSize,
    const int _medianKernelSize)
{
    std::map<std::pair<int, int>, int> grid = computeGrid(_cloud, _leafSize);

    if (grid.empty()) {
        return DepthMapData(); // Return empty matrix if grid is empty
    }

    // 1. Find the min/max grid coordinates to determine map dimensions
    int min_x = grid.begin()->first.first;
    int max_x = grid.begin()->first.first;
    int min_y = grid.begin()->first.second;
    int max_y = grid.begin()->first.second;

    for (const auto& pair : grid) {
        if (pair.first.first < min_x)  min_x = pair.first.first;
        if (pair.first.first > max_x)  max_x = pair.first.first;
        if (pair.first.second < min_y) min_y = pair.first.second;
        if (pair.first.second > max_y) max_y = pair.first.second;
    }

    // 2. Create the depth map, initialized to 0 (or another value for no data)
    cv::Mat depth_map = cv::Mat::zeros(max_y - min_y + 1, max_x - min_x + 1, CV_32FC1);

    // 3. Populate the depth map
    for (const auto& pair : grid) {
        int grid_x = pair.first.first;
        int grid_y = pair.first.second;
        int point_idx = pair.second;

        // Get the depth (Z value) from the original cloud
        float depth = _cloud->points[point_idx].z;

        // Assign it to the correct pixel in the map (after shifting coordinates)
        depth_map.at<float>(grid_y - min_y, grid_x - min_x) = depth;
    }

    // 4. Create a matrix to store the filtered result
    cv::Mat filtered_depth_map;
    cv::medianBlur(depth_map, filtered_depth_map, _medianKernelSize);
    return {filtered_depth_map, grid, min_x, min_y, _leafSize};
}

cv::Point pcToCv(
    const pcl::PointXYZRGB& _point,
    const DepthMapData& _map)
{
    int grid_x = static_cast<int>(std::floor(_point.x / _map.leafSize));
    int grid_y = static_cast<int>(std::floor(_point.y / _map.leafSize));

    int pixel_x = grid_x - _map.min_x;
    int pixel_y = grid_y - _map.min_y;

    return cv::Point(pixel_x, pixel_y);
}

std::map<int, pcl::PointIndices> segmentsToClusters(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const cv::Mat& _markers,
    const DepthMapData& _mapData)
{
    // A map to hold the final clusters, keyed by their segment label.
    std::map<int, pcl::PointIndices> clusters_map;

    // 1. Iterate through EVERY point in the original cloud.
    for (int i = 0; i < _cloud->points.size(); ++i) {
        const auto& point = _cloud->points[i];
        
        // 2. Project the 3D point back to 2D pixel coordinates.
        int grid_x = static_cast<int>(floor(point.x / _mapData.leafSize));
        int grid_y = static_cast<int>(floor(point.y / _mapData.leafSize));
        int u = grid_x - _mapData.min_x; // Pixel column
        int v = grid_y - _mapData.min_y; // Pixel row

        // 3. Check if the point's projection falls within bounds.
        if (v >= 0 && v < _markers.rows && u >= 0 && u < _markers.cols) {
            int label = _markers.at<int>(v, u);

            // Skip background and boundary labels.
            if (label <= 1) {
                continue;
            }

            // 4. Add the point's index directly to the correct cluster.
            //    The map's [] operator creates the entry if it doesn't exist.
            clusters_map[label].indices.push_back(i);
        }
    }

    return clusters_map;
}

std::vector<pcl::RGB> generatePclColors(const int _numColors)
{
    std::vector<pcl::RGB> color_table;
    color_table.reserve(_numColors);

    for (int i = 0; i < _numColors; ++i) {
        // Generate a distinct color by cycling through the HSV hue space
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(i * 180.0 / _numColors, 255, 255));
        cv::Mat bgr_mat;
        cv::cvtColor(hsv, bgr_mat, cv::COLOR_HSV2BGR);
        cv::Vec3b bgr = bgr_mat.at<cv::Vec3b>(0, 0);

        // Create a PCL color object with the correct RGB order
        pcl::RGB color;
        color.r = bgr[2]; // Red channel
        color.g = bgr[1]; // Green channel
        color.b = bgr[0]; // Blue channel

        color_table.push_back(color);
    }

    return color_table;
}

std::vector<cv::Vec3b> generateCvColors(const int _numColors)
{
    // Generate a color table with a random BGR color for each segment
    std::vector<cv::Vec3b> colorTab;
    // We add 1 to numSegments because labels are 1-based after connectedComponents
    for (int i = 0; i <= _numColors; ++i) {
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(i * 180.0 / _numColors, 255, 255));
        cv::Mat bgr_mat;
        cv::cvtColor(hsv, bgr_mat, cv::COLOR_HSV2BGR);
        cv::Vec3b bgr = bgr_mat.at<cv::Vec3b>(0, 0);
        colorTab.push_back(cv::Vec3b((uchar)bgr[0], (uchar)bgr[1], (uchar)bgr[2]));
    }

    return colorTab;
}

cv::Mat visualizeMarkers(const cv::Mat& _markers)
{
    // Find the number of segments
    double minVal, maxVal;
    cv::minMaxLoc(_markers, &minVal, &maxVal);
    int numSegments = static_cast<int>(maxVal);
    std::vector<cv::Vec3b> colorTab = generateCvColors(numSegments);

    // Create the visualization image
    cv::Mat markers_viz(_markers.size(), CV_8UC3);

    // Paint the markers image
    for (int i = 0; i < _markers.rows; ++i) {
        for (int j = 0; j < _markers.cols; ++j) {
            int index = _markers.at<int>(i, j);
            if (index == -1) { // Boundaries
                markers_viz.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255); // White
            } else if (index == 0) { // Unknown region
                markers_viz.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);       // Black
            } else { // Object or background segments
                markers_viz.at<cv::Vec3b>(i, j) = colorTab[index];
            }
        }
    }

    return markers_viz;
}

cv::Mat computeGradients(cv::Mat _depthMap, int kernel_size = 5) {
    // 1. Calculate derivatives in x and y directions
    cv::Mat grad_x, grad_y;
    cv::Sobel(_depthMap, grad_x, CV_32F, 1, 0, kernel_size);
    cv::Sobel(_depthMap, grad_y, CV_32F, 0, 1, kernel_size);

    // 2. Calculate the magnitude of the gradient
    cv::Mat gradient_magnitude;
    cv::magnitude(grad_x, grad_y, gradient_magnitude);

    // 3. Normalize to an 8-bit image for watershed input
    cv::Mat gradient_8u;
    cv::normalize(gradient_magnitude, gradient_8u, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 4. Convert to a 3-channel image, which watershed expects
    cv::Mat gradient_for_watershed;
    cv::cvtColor(gradient_8u, gradient_for_watershed, cv::COLOR_GRAY2BGR);

    return gradient_for_watershed;
}

void watershed_markers(
    const cv::Mat& _map,
    const cv::Mat& _sure_bg,
    const cv::Mat& _sure_fg,
    cv::Mat& _markers)
{
    cv::Mat known_markers;
    cv::bitwise_or(_sure_bg, _sure_fg, known_markers);

    cv::Mat unknown;
    cv::bitwise_not(known_markers, unknown);

    cv::connectedComponents(_sure_fg, _markers, 4);
    _markers = _markers + 1;
    _markers.setTo(1, _sure_bg == 255);
    _markers.setTo(0, unknown == 255);
    cv::watershed(_map, _markers);
}

cv::Mat computeTopHat(const cv::Mat& _map, const float _xFactor = 10.0, const int _kernelSize=5)
{
    cv::Mat ellipse_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(_kernelSize, _kernelSize));
    cv::Mat ridges;
    cv::morphologyEx(_map, ridges, cv::MORPH_TOPHAT, ellipse_kernel);

    cv::Mat landscape = _map + ridges * _xFactor;

    return landscape;
}

cv::Mat invertFloatMap(const cv::Mat& _map)
{
    cv::Mat inverted_map;
    double min_val, max_val;
    cv::minMaxLoc(_map, &min_val, &max_val, 0);
    cv::subtract(cv::Scalar::all(max_val), _map, inverted_map);

    return inverted_map;
}

cv::Mat growSeedsWithinBoundaries(
    const cv::Mat& _input_mask,
    const cv::Mat& _seeds,
    const int _growth_kernel=7)
{
    // 1. Find and label all independent blobs in the input mask.
    cv::Mat labels;
    int num_labels = cv::connectedComponents(_input_mask, labels, 4);
    
    // Create the final output canvas.
    cv::Mat final_grown_mask = cv::Mat::zeros(_input_mask.size(), CV_8U);

    if (num_labels < 2) return final_grown_mask;

    // 2. Loop through each blob and process it in complete isolation.
    for (int i = 1; i < num_labels; ++i) {
        // --- Isolate Territory ---
        // Create a mask for the current blob's territory only.
        cv::Mat parent_mask = (labels == i);

        // --- Isolate the Corresponding Seed ---
        // Use bitwise_and to find the seed(s) from _seeds that are inside this territory.
        cv::Mat current_seed;
        cv::bitwise_and(_seeds, parent_mask, current_seed);

        // --- Grow ---
        // Grow the isolated seed, but confine it ONLY to its own parent's territory.
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(_growth_kernel, _growth_kernel));
        cv::dilate(current_seed, current_seed, kernel);
        cv::bitwise_and(current_seed, parent_mask, current_seed);
        
        // 3. Add the result for this blob to the final mask.
        cv::bitwise_or(final_grown_mask, current_seed, final_grown_mask);
    }

    return final_grown_mask;
}

void fusePacManBlobs(
    cv::Mat& _markers,
    const double _solidity_threshold = 0.9)
{
    std::cout << "fusePacManBlobs" << std::endl;
    // --- FIX 1: Use a new 'labels' matrix to avoid overwriting input ---
    // We get stats from the binary version, but keep original labels for later.
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(_markers > 1, labels, stats, centroids, 4);

    if (num_labels < 3) return; // Need at least two blobs to compare.

    std::vector<std::vector<cv::Point>> contours(num_labels);
    std::vector<std::vector<cv::Point>> hulls(num_labels);
    std::vector<double> solidity(num_labels);
    std::vector<cv::Point> safe_points(num_labels);

    // Pre-calculate contours, hulls, and solidity for each blob.
    for (int i = 1; i < num_labels; ++i) {
        // Use the temporary 'labels' matrix to get each component's mask
        cv::Mat blob_mask = (labels == i);
        std::vector<std::vector<cv::Point>> blob_contours;
        cv::findContours(blob_mask, blob_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!blob_contours.empty()) {
            contours[i] = blob_contours[0];
            double area = cv::contourArea(contours[i]);
            cv::convexHull(contours[i], hulls[i]);
            double hull_area = cv::contourArea(hulls[i]);
            if (hull_area > 0) {
                solidity[i] = area / hull_area;
            }
        }

        cv::Mat dist_transform;
        cv::distanceTransform(blob_mask, dist_transform, cv::DIST_L2, 5);
        cv::minMaxLoc(dist_transform, nullptr, nullptr, nullptr, &safe_points[i]);
    }

    std::map<int, int> remap_table;

    for (int i = 1; i < num_labels; ++i) {
        for (int j = i + 1; j < num_labels; ++j) {

            // --- NEW LOGIC: Determine eater based on size (pixel area) ---
            int area_i = stats.at<int>(i, cv::CC_STAT_AREA);
            int area_j = stats.at<int>(j, cv::CC_STAT_AREA);

            int eater_idx, eaten_idx;
            if (area_i >= area_j) {
                eater_idx = i;
                eaten_idx = j;
            } else {
                eater_idx = j;
                eaten_idx = i;
            }

            // Get the original labels from the input _markers Mat using the safe points
            int eater_label = _markers.at<int>(safe_points[eater_idx]);
            int eaten_label = _markers.at<int>(safe_points[eaten_idx]);
            
            if (eater_label <= 1 || eaten_label <= 1 || eater_label == eaten_label) continue;
            
            // Get centroids and check if hulls are valid
            cv::Point2d eater_centroid(centroids.at<double>(eater_idx, 0), centroids.at<double>(eater_idx, 1));
            cv::Point2d eaten_centroid(centroids.at<double>(eaten_idx, 0), centroids.at<double>(eaten_idx, 1));
            
            if (hulls[eater_idx].empty() || hulls[eaten_idx].empty()) continue;
            
            // --- Proximity Check ---
            // We still need to check if the blobs are close to each other.
            // A simple way is to check if either centroid is inside the other's convex hull.
            double dist_eaten_in_eater_hull = cv::pointPolygonTest(hulls[eater_idx], eaten_centroid, true);
            double dist_eater_in_eaten_hull = cv::pointPolygonTest(hulls[eaten_idx], eater_centroid, true);
            
            // If they are close enough, perform the fusion.
            if (dist_eaten_in_eater_hull >= -1.0 || dist_eater_in_eaten_hull >= -1.0) {
                std::cout << "Fusing blob " << eaten_label << " into blob " << eater_label << " based on size." << std::endl;
                remap_table[eaten_label] = eater_label;
            }
        }
    }

    // --- FIX 3: Resolve chained fusions (e.g., A->B, B->C becomes A->C) ---
    bool changed = true;
    while(changed) {
        changed = false;
        for (auto& pair : remap_table) {
            if (remap_table.count(pair.second)) {
                pair.second = remap_table[pair.second];
                changed = true;
            }
        }
    }

    // Apply the fusion by re-labeling the markers image.
    std::set<int> remapped_labels;
    for (int y = 0; y < _markers.rows; ++y) {
        for (int x = 0; x < _markers.cols; ++x) {
            int& label = _markers.at<int>(y, x);
            if (remap_table.count(label)) {
                label = remap_table[label];
                if(label > 1) {
                    remapped_labels.insert(label);
                }
            }
        }
    }

    // For each unique blob, apply closing to fill its internal gaps.
    for (int label : remapped_labels) {
        // Create a mask for the current blob only.
        cv::Mat blob_mask = (_markers == label);

        // Close the gaps in the mask.
        cv::morphologyEx(blob_mask, blob_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

        // Update the main markers image with the filled blob.
        _markers.setTo(label, blob_mask);
    }
}

cv::Mat keepLargestBlob(const cv::Mat& _mask)
{
    // 1. Find all connected components and their stats
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(_mask, labels, stats, centroids, 4);

    // If there are 2 labels, it means 1 background (0) and 1 blob (1). No need to process.
    if (num_labels < 3) {
        return _mask.clone();
    }

    // 2. Find the label of the largest component (excluding the background label 0)
    int largest_area = 0;
    int largest_label = 0;

    // Start from label 1 to ignore the background
    for (int i = 1; i < num_labels; ++i) {
        int current_area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (current_area > largest_area) {
            largest_area = current_area;
            largest_label = i;
        }
    }

    // 3. Create a new mask containing only the largest blob
    cv::Mat largest_blob_mask = (labels == largest_label);

    return largest_blob_mask;
}

void removeOutgrowths(cv::Mat& _markers)
{
    std::set<int> unique_labels;
    for (int y = 0; y < _markers.rows; ++y) {
        for (int x = 0; x < _markers.cols; ++x) {
            int& label = _markers.at<int>(y, x);
            if(label > 1) {
                unique_labels.insert(label);
            }
        }
    }

    for (int label : unique_labels) {
        // Create a mask for the current blob only.
        cv::Mat blob_mask = (_markers == label);

        cv::Mat cleaned_mask;
        cv::morphologyEx(blob_mask, cleaned_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        cleaned_mask = keepLargestBlob(cleaned_mask);

        cv::Mat outgrowth_mask = blob_mask - cleaned_mask;
        _markers.setTo(-1, outgrowth_mask);
    }
}

std::vector<pcl::PointIndices> clustersMapToVector(
    const std::map<int, pcl::PointIndices>& _clusters_map)
{
    std::vector<pcl::PointIndices> clusters_vector;
    // Reserve space for efficiency
    clusters_vector.reserve(_clusters_map.size());

    // Iterate through the map and push the value (pair.second) into the vector
    for (const auto& pair : _clusters_map) {
        clusters_vector.push_back(pair.second);
    }

    return clusters_vector;
}

int getSegmentLabelForPoint(
    const pcl::PointXYZRGB& _point,
    const cv::Mat& _markers,
    const DepthMapData& _mapData)
{
    // 1. Project the 3D point to 2D pixel coordinates.
    int u = static_cast<int>(std::floor(_point.x / _mapData.leafSize)) - _mapData.min_x;
    int v = static_cast<int>(std::floor(_point.y / _mapData.leafSize)) - _mapData.min_y;

    // 2. Clamp coordinates to ensure they are within the image bounds.
    int u_clamped = std::max(0, std::min(_markers.cols - 1, u));
    int v_clamped = std::max(0, std::min(_markers.rows - 1, v));

    // 3. Check the label at the point's direct (clamped) projection.
    int label = _markers.at<int>(v_clamped, u_clamped);
    if (label > 1) {
        return label;
    }

    // 4. If the label is not valid, search outwards in an expanding box.
    // The search radius 'd' increases on each iteration.
    int max_search_dist = std::max(_markers.rows, _markers.cols);
    for (int d = 1; d < max_search_dist; ++d) {
        // Iterate over the perimeter of a square with side length 2*d + 1
        for (int i = -d; i <= d; ++i) {
            for (int j = -d; j <= d; ++j) {
                // Skip the inner part of the box, which has already been checked.
                if (std::abs(i) != d && std::abs(j) != d) {
                    continue;
                }

                int nu = u_clamped + j;
                int nv = v_clamped + i;

                // Check if the neighbor pixel is within the image bounds.
                if (nv >= 0 && nv < _markers.rows && nu >= 0 && nu < _markers.cols) {
                    label = _markers.at<int>(nv, nu);
                    if (label > 1) {
                        return label; // Found the closest valid neighbor.
                    }
                }
            }
        }
    }

    // If no valid label was found anywhere, return -1.
    return -1;
}

std::vector<pcl::PointIndices> computeWatershed(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const float _leafSize,
    const float _radius,
    const float _smooth_factor,
    const int _medianKernelSize,
    const int _tophat_kernel,
    const float _tophat_amplification,
    const float _pacman_solidity,
    const bool _shouldView,
    std::optional<pcl::PointXYZRGB> _point)
{
    bool has_landing_point = false;
    pcl::PointXYZRGB point;
    if(_point){
        point = _point.value();
        has_landing_point = true;
    };

    const float SMOOTH_RADIUS = _smooth_factor*_radius;
    const float EXTREMUMS_RADIUS = _radius;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    smoothPC(smoothCloud, SMOOTH_RADIUS);

    DepthMapData depthMapData = computeDepthMap(_cloud, _leafSize, _medianKernelSize);
    DepthMapData smoothDepthMapData = computeDepthMap(smoothCloud, _leafSize, _medianKernelSize);

    // Define "sure background" as all pixels with 0 value in the original depth map
    cv::Mat sure_bg;
    cv::compare(smoothDepthMapData.depthMap, 0, sure_bg, cv::CMP_EQ);

    // Create an inverted version of the background mask
    cv::Mat foreground_mask;
    cv::bitwise_not(sure_bg, foreground_mask);

    // --- Method 2: Get seeds from 3D Local Extrema ---
    pcl::PointIndices maximumIdx = pcl_tools::findLocalExtremums(smoothCloud, EXTREMUMS_RADIUS, false);
    cv::Mat sure_fg_extremums = cv::Mat::zeros(smoothDepthMapData.depthMap.size(), CV_8UC1);

    for (const int& index : maximumIdx.indices) {
        const auto& cloudPoint = smoothCloud->points[index];

        // Project the 3D point back to 2D grid/pixel coordinates.
        int grid_x = static_cast<int>(floor(cloudPoint.x / _leafSize));
        int grid_y = static_cast<int>(floor(cloudPoint.y / _leafSize));

        int u = grid_x - smoothDepthMapData.min_x; // Pixel column
        int v = grid_y - smoothDepthMapData.min_y; // Pixel row

        // Check if the projection is within the image bounds.
        if (v >= 0 && v < sure_fg_extremums.rows && u >= 0 && u < sure_fg_extremums.cols) {
            // Set the corresponding pixel to white (255) to mark it as sure foreground.
            sure_fg_extremums.at<uchar>(v, u) = 255;
        }
    }

    // --- Method 1: Get seeds from Distance Transform ---
    // 1. Invert the marker mask so markers are 0 for the distance transform.
    cv::Mat inverted_extremums;
    cv::bitwise_not(sure_fg_extremums, inverted_extremums);

    // 2. Calculate the distance transform. This correctly creates basins at the markers.
    cv::Mat dist_map;
    cv::distanceTransform(inverted_extremums, dist_map, cv::DIST_L2, 5);

    cv::Mat flat_basins_mask = dist_map < 2;
    dist_map.setTo(0, flat_basins_mask);

    cv::Mat landscape = computeTopHat(dist_map, _tophat_amplification, _tophat_kernel);

    // 4. Prepare for watershed. NO INVERSION STEP IS NEEDED.
    cv::Mat dist_map_norm;
    cv::normalize(landscape, dist_map_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    dist_map_norm.setTo(255, sure_bg);

    cv::Mat bgr_dist_map;
    cv::cvtColor(dist_map_norm, bgr_dist_map, cv::COLOR_GRAY2BGR);

    // 5. Label the original markers and run the watershed.
    cv::Mat dist_markers;
    watershed_markers(bgr_dist_map, sure_bg, sure_fg_extremums, dist_markers);

    cv::Mat separated_blobs = dist_markers > 1;

    cv::Mat sure_fg = growSeedsWithinBoundaries(separated_blobs, sure_fg_extremums, 5);

    // 1. NORMALIZE AND COLORIZE THE DEPTH MAP
    // Convert the 32F depth map to a visual 8U format
    cv::Mat inverted_depth;
    inverted_depth = invertFloatMap(smoothDepthMapData.depthMap);

    cv::Mat hat_inverted_depth = computeTopHat(inverted_depth, _tophat_amplification, _tophat_kernel);

    cv::Mat inverted_norm_depth;
    cv::normalize(hat_inverted_depth, inverted_norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1, foreground_mask);
    inverted_norm_depth.setTo(255, sure_bg);

    // Create a 3-channel color image for the watershed algorithm
    cv::Mat color_depth;
    cv::cvtColor(inverted_norm_depth, color_depth, cv::COLOR_GRAY2BGR);

    cv::Mat markers;
    watershed_markers(color_depth, sure_bg, sure_fg, markers);

    cv::Mat markers_b4_pacman = markers.clone();
    fusePacManBlobs(markers, _pacman_solidity);

    cv::Mat markers_b4_outgrowths = markers.clone();
    removeOutgrowths(markers);

    std::map<int, pcl::PointIndices> clusters_map = segmentsToClusters(_cloud, markers, smoothDepthMapData);
    std::vector<pcl::PointIndices> clusters_vec = clustersMapToVector(clusters_map);

    if(_shouldView){
        // 5. VISUALIZE THE RESULT
        cv::Mat dist_markers_viz = visualizeMarkers(dist_markers);
        cv::Mat markers_b4_pacman_viz = visualizeMarkers(markers_b4_pacman);
        cv::Mat markers_b4_outgrowths_viz = visualizeMarkers(markers_b4_outgrowths);
        cv::Mat markers_viz = visualizeMarkers(markers);
        
        // Blend the result with the original color depth map
        cv::Mat bgr_dist_map_viz;
        cv::Mat color_depth_viz;

        cv::applyColorMap(bgr_dist_map, bgr_dist_map_viz, cv::COLORMAP_JET);
        cv::applyColorMap(color_depth, color_depth_viz, cv::COLORMAP_JET);

        cv::Vec3b point_color(0, 0, 0); 

        cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Background", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Foreground extremums", cv::WINDOW_NORMAL);
        cv::namedWindow("Color Dist Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers distance", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Foreground", cv::WINDOW_NORMAL);
        cv::namedWindow("Color Depth Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers before pacman", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers before outgrowths", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers", cv::WINDOW_NORMAL);

        // You can also show intermediate steps
        cv::imshow("Depth Map", inverted_norm_depth);
        cv::imshow("Sure Background", sure_bg);
        cv::imshow("Sure Foreground extremums", sure_fg_extremums);
        cv::imshow("Color Dist Map", bgr_dist_map_viz);
        cv::imshow("Markers distance", dist_markers_viz);
        cv::imshow("Sure Foreground", sure_fg);
        cv::imshow("Color Depth Map", color_depth_viz);
        cv::imshow("Markers before pacman", markers_b4_pacman_viz);
        cv::imshow("Markers before outgrowths", markers_b4_outgrowths_viz);

        if(has_landing_point){
            cv::Point cv_point = pcToCv(point, smoothDepthMapData);
            markers_viz.at<cv::Vec3b>(cv_point.y, cv_point.x) = point_color;
        }
        cv::imshow("Markers", markers_viz);

        cv::waitKey(0);

        std::vector<pcl::RGB> colorTable = generatePclColors(clusters_vec.size());
        colorSegmentedPoints(_cloud, pcl::RGB(255,255,255));
        for(size_t i=0; i < clusters_vec.size(); ++i) {
            colorSegmentedPoints(_cloud, clusters_vec[i], colorTable[i]);
        }

        for (auto& point : *_cloud)
        {
            if (point.r == 255 && 
                point.g == 255 && 
                point.b == 255)
            {
                point.r = 255;
                point.g = 255;
                point.b = 0;
            }
        }
    }
    
    // if(has_landing_point){
    //     return extractClosestCluster(
    //         _cloud,
    //         clusters_vec,
    //         point
    //     );
    // }
    // else{
    //     auto biggest_cluster_it = std::max_element(
    //         clusters_vec.begin(),
    //         clusters_vec.end(),
    //         [](const auto& a, const auto& b) {
    //             return a.indices.size() < b.indices.size();
    //         }
    //     );
    //     return extractPoints<pcl::PointXYZRGB>(_cloud, biggest_cluster_it->second);
    // }

    return clusters_vec;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extractClusters(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const std::vector<pcl::PointIndices>& _clusters)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extracted_clusters;
    extracted_clusters.reserve(_clusters.size());

    for(const auto& cluster : _clusters)
    {
        extracted_clusters.emplace_back(extractPoints<pcl::PointXYZRGB>(_cloud, cluster));
    }

    return extracted_clusters;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateGridCloudFromCenter(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const OrientedBoundingBox& _treeBB,
    const float& _radius,
    const float& _radius_factor,
    const float _max_ratio_from_center)
{
    float step = _radius_factor*_radius;

    // Check for zero spacing to prevent an infinite loop.
    if (step <= 0) {
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(); // Return empty cloud if spacing is invalid
    }

    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*_treeCloud, min_pt, max_pt);
    float min_x = min_pt.x + _radius/2.0;
    float max_x = max_pt.x - _radius/2.0;
    float min_y = min_pt.y + _radius/2.0;
    float max_y = max_pt.y - _radius/2.0;

    pcl::PointXYZRGB treeCenterPoint(_treeBB.centroid[0], _treeBB.centroid[1], 0.0, 255, 255, 255);
    projectPoint(_treeCloud, treeCenterPoint);

    pcl::PointXYZRGB treeHighestPoint = getHighestPoint(_treeCloud);

    pcl::PointXYZRGB treeMidwayPoint;
    treeMidwayPoint.x = (treeCenterPoint.x + treeHighestPoint.x) / 2.0f;
    treeMidwayPoint.y = (treeCenterPoint.y + treeHighestPoint.y) / 2.0f;
    projectPoint(_treeCloud, treeMidwayPoint);

    std::vector<std::pair<pcl::PointXYZRGB,double>> candidate_points;
    for (float x = min_x; x <= max_x; x += step) {
        for (float y = min_y; y <= max_y; y += step) {
            pcl::PointXYZRGB point;
            point.x = x;
            point.y = y;
            point.z = projectPoint(_treeCloud, point);

            float dist3D = computePointsDist3D(point, treeMidwayPoint);
            float min_diameter = std::min(_treeBB.width, _treeBB.height);
            float ratio3D = dist3D/min_diameter;

            if(ratio3D < _max_ratio_from_center) {
                candidate_points.push_back({point, ratio3D});
            }
        }
    }

    std::sort(candidate_points.begin(), candidate_points.end(), 
        [](const std::pair<pcl::PointXYZRGB,double>& a, const std::pair<pcl::PointXYZRGB,double>& b) {
            return a.second < b.second;
    });

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr gridCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    gridCloud->reserve(candidate_points.size()); // Optimize memory allocation
    for (const auto& candidate : candidate_points) {
        gridCloud->emplace_back(candidate.first);
    }

    std::cout << "gridCloud has " << gridCloud->points.size() << " points." << std::endl;

    return gridCloud;
}

/**
 * @brief Calculates the shortest distance from a point to a polygon's perimeter.
 * @param point The PointXYZRGB to test.
 * @param polygon The polygon, represented by a cloud of ordered PointXYZRGB vertices.
 * @return The minimum Euclidean distance to the polygon's edge.
 */
double pointToPolygonDistance(const pcl::PointXYZRGB& _point, const pcl::PointCloud<pcl::PointXYZRGB>& _polygon)
{
    double min_dist_sq = std::numeric_limits<double>::max();
    int n_points = _polygon.points.size();

    // Iterate over each edge of the polygon (from point i to point j)
    for (int i = 0, j = n_points - 1; i < n_points; j = i++)
    {
        const auto& p_i = _polygon.points[i];
        const auto& p_j = _polygon.points[j];

        double edge_dx = p_j.x - p_i.x;
        double edge_dy = p_j.y - p_i.y;

        double dist_sq;

        if (edge_dx == 0.0 && edge_dy == 0.0) // Edge has zero length
        {
            dist_sq = (_point.x - p_i.x) * (_point.x - p_i.x) + (_point.y - p_i.y) * (_point.y - p_i.y);
        }
        else
        {
            // Project point onto the line defined by the edge
            double t = ((_point.x - p_i.x) * edge_dx + (_point.y - p_i.y) * edge_dy) / (edge_dx * edge_dx + edge_dy * edge_dy);

            if (t < 0.0) // Projection is before the start of the segment
            {
                dist_sq = (_point.x - p_i.x) * (_point.x - p_i.x) + (_point.y - p_i.y) * (_point.y - p_i.y);
            }
            else if (t > 1.0) // Projection is after the end of the segment
            {
                dist_sq = (_point.x - p_j.x) * (_point.x - p_j.x) + (_point.y - p_j.y) * (_point.y - p_j.y);
            }
            else // Projection is on the segment
            {
                double closest_x = p_i.x + t * edge_dx;
                double closest_y = p_i.y + t * edge_dy;
                dist_sq = (_point.x - closest_x) * (_point.x - closest_x) + (_point.y - closest_y) * (_point.y - closest_y);
            }
        }
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
        }
    }
    return std::sqrt(min_dist_sq);
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr generateGridCloudFromEdge(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeEdge,
    const float& _radius,
    const float& _radius_factor,
    const float _max_dist_from_edge)
{
    float step = _radius_factor*_radius;

    // Check for zero spacing to prevent an infinite loop.
    if (step <= 0) {
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(); // Return empty cloud if spacing is invalid
    }

    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*_treeCloud, min_pt, max_pt);
    float min_x = min_pt.x + _radius/2.0;
    float max_x = max_pt.x - _radius/2.0;
    float min_y = min_pt.y + _radius/2.0;
    float max_y = max_pt.y - _radius/2.0;

    std::cout << "min_x: " << min_x << std::endl;
    std::cout << "max_x: " << max_x << std::endl;
    std::cout << "min_y: " << min_y << std::endl;
    std::cout << "max_y: " << max_y << std::endl;

    std::vector<std::pair<pcl::PointXYZRGB,double>> candidate_points;
    for (float x = min_x; x <= max_x; x += step) {
        for (float y = min_y; y <= max_y; y += step) {
            pcl::PointXYZRGB point;
            point.x = x;
            point.y = y;
            point.z = projectPoint(_treeCloud, point);

            bool is_inside = pcl::isPointIn2DPolygon(point, *_treeEdge);
            float sign = is_inside ? -1.0 : 1.0;
            // double dist = pointToPolygonDistance(point, *_treeEdge);
            double dist;

            std::vector<float> distances2D;
            distances2D.reserve(_treeEdge->points.size());
            for(const auto& edgePoint : _treeEdge->points) {
                float dx = point.x - edgePoint.x;
                float dy = point.y - edgePoint.y;

                distances2D.push_back(std::sqrt(dx*dx + dy*dy));
            }

            auto min_dist2D_it = std::min_element(distances2D.begin(), distances2D.end());
            if (min_dist2D_it != distances2D.end()) {
                dist = sign * (*min_dist2D_it);
            }

            // std::cout << "is_inside: " << is_inside << std::endl;
            // std::cout << "dist: " << dist << std::endl;

            if(dist < _max_dist_from_edge) {
                candidate_points.push_back({point, dist});
            }
        }
    }

    std::sort(candidate_points.begin(), candidate_points.end(), 
        [](const std::pair<pcl::PointXYZRGB,double>& a, const std::pair<pcl::PointXYZRGB,double>& b) {
            return a.second < b.second;
    });

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr gridCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    gridCloud->reserve(candidate_points.size()); // Optimize memory allocation
    for (const auto& candidate : candidate_points) {
        gridCloud->emplace_back(candidate.first);
    }

    std::cout << "gridCloud has " << gridCloud->points.size() << " points." << std::endl;

    return gridCloud;
}

double distanceToBBSq(const pcl::PointXYZRGB& _point, const pcl_tools::BoundingBox& _bbox, bool is_2d = true)
{
    // First, check if the point is inside the bounding box
    bool is_inside_2d = (_point.x >= _bbox.min_x && _point.x <= _bbox.max_x &&
                         _point.y >= _bbox.min_y && _point.y <= _bbox.max_y);
    
    bool is_inside_3d = is_inside_2d && (_point.z >= _bbox.min_z && _point.z <= _bbox.max_z);

    bool is_inside = is_2d ? is_inside_2d : is_inside_3d;

    if (is_inside) {
        // --- If inside, distance is to the centroid ---
        Eigen::Vector3f diff = _bbox.centroid - _point.getVector3fMap();
        if (is_2d) {
            // Ignore the Z component for 2D distance
            return diff.x() * diff.x() + diff.y() * diff.y();
        } else {
            return diff.squaredNorm();
        }
    } else {
        // --- If outside, distance is to the surface (original logic) ---
        float dx = std::max({0.0f, _bbox.min_x - _point.x, _point.x - _bbox.max_x});
        float dy = std::max({0.0f, _bbox.min_y - _point.y, _point.y - _bbox.max_y});

        if (is_2d) {
            return dx * dx + dy * dy;
        } else {
            float dz = std::max({0.0f, _bbox.min_z - _point.z, _point.z - _bbox.max_z});
            return dx * dx + dy * dy + dz * dz;
        }
    }
}

DistsOfInterest computeDistancesToPolygon(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _hull_polygon,
    const pcl_tools::OrientedBoundingBox& _treeBB,
    const DistsOfInterest& _distsOfInterest)
{
    DistsOfInterest output = _distsOfInterest;
    double min_tree_dia = std::min(_treeBB.width, _treeBB.height);
    double min_tree_radius = min_tree_dia / 2.0;

    if (_hull_polygon->points.size() < 3) {
        return output;
    }

    bool is_inside = pcl::isPointIn2DPolygon(_landingPoint, *_hull_polygon);
    double sign = is_inside ? -1.0 : 1.0;
    // double magnitude = sign * pointToPolygonDistance(_landingPoint, *_hull_polygon);
    // output.distMinBoundary2D = magnitude;

    std::vector<float> distances2D;
    distances2D.reserve(_hull_polygon->points.size());
    for(const auto& point : _hull_polygon->points) {
        float dx = _landingPoint.x - point.x;
        float dy = _landingPoint.y - point.y;
        float dz = _landingPoint.z - point.z;

        distances2D.push_back(std::sqrt(dx*dx + dy*dy));
    }

    auto min_dist2D_it = std::min_element(distances2D.begin(), distances2D.end());
    if (min_dist2D_it != distances2D.end()) {
        output.distMinBoundary2D = sign * (*min_dist2D_it);
    }

    output.ratioMinBoundary2D = output.distMinBoundary2D / min_tree_radius;

    return output;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractClosestCluster(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const std::vector<pcl::PointIndices>& _clusters,
    const pcl::PointXYZRGB& _point,
    const double _alpha)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extracted_clusters = extractClusters(_cloud, _clusters);

    if (extracted_clusters.empty()) {
        return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr closest_cluster = nullptr;
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& cluster : extracted_clusters)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_polygon = pcl_tools::computeConcaveHull2D(
            cluster,
            _alpha
        );

        OrientedBoundingBox obb = getOBB(cluster);
        DistsOfInterest distsOfInterest = computeDistancesToPolygon(
            _point,
            hull_polygon,
            obb,
            DistsOfInterest()
        );

        if (distsOfInterest.distMinBoundary2D < min_dist) {
            min_dist = distsOfInterest.distMinBoundary2D;
            closest_cluster = cluster;
        }
    }

    return closest_cluster;
}

void smoothPC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud, const float _searchRadius)
{
    // Output has the PointNormal type in order to store the normals calculated by MLS
    pcl::PointCloud<pcl::PointNormal> mls_points;

    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZRGB, pcl::PointNormal> mls;
    
    mls.setComputeNormals(true);

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_pointCloud);
    // Set parameters
    mls.setInputCloud(_pointCloud);
    mls.setPolynomialOrder(7);
    mls.setSearchMethod(kdtree);
    mls.setSearchRadius(_searchRadius);

    // Reconstruct
    mls.process(mls_points);

    // Manual copy because of different types PointNormal vs PointXYZRGB
    for (size_t i = 0; i < mls_points.size(); ++i) {
        _pointCloud->points[i].x = mls_points.points[i].x;
        _pointCloud->points[i].y = mls_points.points[i].y;
        _pointCloud->points[i].z = mls_points.points[i].z;
    }

    // pcl::copyPointCloud(mls_points, *_pointCloud);
}

void removeNoise(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const int _nNeighbors,
    const float _stdDev,
    const bool _isNegative)
{
    // const float densityFactor = _pointCloud->size() / (_pointCloud->width * _pointCloud->height);
    // const int nNeighbors = std::clamp(static_cast<int>(30 * densityFactor), 10, 50);
    // const float stdDev = 1.0 + (0.5 * (1.0 - densityFactor)); // 1.0-1.5σ

    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(_pointCloud);
    sor.setMeanK(_nNeighbors);      // Analyze 50 nearest neighbors
    sor.setStddevMulThresh(_stdDev);    // Remove points >1 standard deviation from mean
    sor.setNegative(_isNegative);         // Keep inliers (set true to visualize outliers)
    sor.filter(*_pointCloud);
}

void thresholdPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const std::string& _filterFieldName,
    const float _thresholdMin,
    const float _thresholdMax)
{
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(_pointCloud);
    pass.setFilterFieldName(_filterFieldName);
    pass.setFilterLimits(_thresholdMin, _thresholdMax);
    pass.filter(*_pointCloud);
}

pcl::PointIndices findBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const int _searchNeighbors)
{
    pcl::BoundaryEstimation<pcl::PointXYZRGB, pcl::PointNormal, pcl::Boundary> be;
    pcl::PointCloud<pcl::Boundary>::Ptr boundaryCloud(new pcl::PointCloud<pcl::Boundary>);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(_cloud);

    be.setInputCloud(_cloud);
    be.setInputNormals(_normalsCloud);
    be.setSearchMethod(kdTree);
    be.setKSearch(_searchNeighbors); // Adjust K value as necessary
    be.compute(*boundaryCloud);

    pcl::PointIndices boundaryIdx;
    boundaryIdx.indices.reserve(boundaryCloud->points.size());
    for (size_t i = 0; i < boundaryCloud->points.size(); ++i)
    {
        uint8_t x = static_cast<int>(boundaryCloud->points[i].boundary_point);
        if (x == 1) // Boundary point
        {
            boundaryIdx.indices.emplace_back(i);
        }
    }
    boundaryIdx.indices.shrink_to_fit();

    return boundaryIdx;
}

void extractBoundary(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const int _searchNeighbors)
{
    pcl::PointIndices idx = findBoundary(
        _cloud,
        _normalsCloud,
        _searchNeighbors
    );

    std::cout << "extractBoundary" << std::endl;
    std::cout << "idx length: " << idx.indices.size() << std::endl;

    _cloud = extractPoints<pcl::PointXYZRGB>(_cloud, idx, false);
}

pcl::PointIndices findRadiusBoundary(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointIndices _boundaryIdx,
    const float _searchRadius)
{
    std::unordered_set<int> unique_indices;
    unique_indices.reserve(_cloud->size());

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdTree->setInputCloud(_cloud);

    std::vector<int> point_idx_radius_search;
    std::vector<float> point_radius_squared_distance;

    // For each boundary point, find its neighbors.
    for (const int idx : _boundaryIdx.indices)
    {
        if (kdTree->radiusSearch(_cloud->points[idx], _searchRadius, point_idx_radius_search, point_radius_squared_distance) > 0)
        {
            // Insert all found neighbors into the set.
            // Duplicates will be ignored automatically.
            unique_indices.insert(point_idx_radius_search.begin(), point_idx_radius_search.end());
        }
    }

    // Create the final PointIndices object and copy the unique indices into it.
    pcl::PointIndices pointsIdxWithinRadius;
    pointsIdxWithinRadius.indices.assign(unique_indices.begin(), unique_indices.end());

    return pointsIdxWithinRadius;
}

void extractRadiusBoundary(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const pcl::PointIndices _boundaryIdx,
    const float _searchRadius,
    const bool _should_invert)
{
    pcl::PointIndices idx = findRadiusBoundary(
        _cloud,
        _boundaryIdx,
        _searchRadius
    );

    _cloud = extractPoints<pcl::PointXYZRGB>(_cloud, idx, _should_invert);
}

pcl::PointIndices findSurface(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _leafSize)
{
    std::map<std::pair<int, int>, int> grid = computeGrid(_cloud, _leafSize);

    pcl::PointIndices surfacePointsIdx;
    surfacePointsIdx.indices.reserve(grid.size());

    for (const auto& element : grid) {
        surfacePointsIdx.indices.emplace_back(element.second);
    }
    surfacePointsIdx.indices.shrink_to_fit();

    return surfacePointsIdx;
}

void extractSurface(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const float _leafSize)
{
    pcl::PointIndices idx = findSurface(
        _cloud,
        _leafSize
    );

    _cloud = extractPoints<pcl::PointXYZRGB>(_cloud, idx, false);
}

float computeDensity(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const float _radius)
{
    // Compute the volume of the sphere
    float volume = (4.0f / 3.0f) * M_PI * std::pow(_radius, 3);

    // Compute the density
    float density = _cloud->points.size() / volume;

    std::cout << "Density: " << density << std::endl;
    return density;
}

float computeSurfaceDensity(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _radius)
{
    // Compute the area of the circle
    float area = M_PI * std::pow(_radius, 2);

    // Compute the density
    float density = _cloud->points.size() / area;

    std::cout << "Density: " << density << std::endl;
    return density;
}

float projectPoint(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& _cloud,
    pcl::PointXYZRGB& _point,
    const int _numNeighbors)
{
    // Validate inputs
    if (!_cloud || _cloud->empty() || _numNeighbors <= 0) {
        _point.z = 0.0f; // Set to a default value
        return 0.0f;
    }

    // 1. Create a 2D representation of the cloud for an XY-plane search.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_2d(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    for (auto& point : cloud_2d->points) {
        point.z = 0;
    }

    // 2. Set up the KdTree for 2D search
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cloud_2d);

    // 3. Create a 2D search point (ignoring the original Z) to ensure a true 2D search
    pcl::PointXYZRGB search_point_2d(_point.x, _point.y, 0.0f, 0, 0, 0);

    // 4. Perform the nearest neighbor search
    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_distances;
    float median_z = 0.0f; // Default value if no neighbors are found

    if (kdtree->nearestKSearch(search_point_2d, _numNeighbors, neighbor_indices, neighbor_distances) > 0)
    {
        // 5. Collect all Z values from the found neighbors
        std::vector<float> z_values;
        z_values.reserve(neighbor_indices.size());
        for (const int& index : neighbor_indices) {
            z_values.emplace_back(_cloud->points[index].z);
        }

        // 6. Calculate the median of the Z values
        if (!z_values.empty()) {
            // Sort the values to find the middle one
            std::sort(z_values.begin(), z_values.end());

            size_t n = z_values.size();
            if (n % 2 == 0) {
                // If the count is even, average the two middle elements
                median_z = (z_values[n / 2 - 1] + z_values[n / 2]) / 2.0f;
            } else {
                // If the count is odd, pick the single middle element
                median_z = z_values[n / 2];
            }
        }
    }

    // 7. Update the Z coordinate of the input point and return the median
    _point.z = median_z;
    return median_z;
}

pcl::PrincipalCurvatures computeCurvature(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointXYZRGB& _point,
    const float _radius)
{
    if (_cloud->empty()) {
        // throw std::runtime_error("Input cloud for computeCurvature cannot be empty.");
        std::cout << "Error: Input cloud for computeCurvature cannot be empty." << std::endl;
        return pcl::PrincipalCurvatures();
    }

    // Instead of copying the whole cloud, create a small cloud containing only the neighbors.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr curvatureCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));

    // Add the query point itself to the neighborhood cloud.
    curvatureCloud->push_back(_point);
    const int target_idx = curvatureCloud->size() - 1;

    // Step 3: Compute normals for the whole cloud
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormalsRad(curvatureCloud, _radius);

    Eigen::Vector3f target_normal(normals->points[target_idx].normal_x,
                                  normals->points[target_idx].normal_y,
                                  normals->points[target_idx].normal_z);

    // For a consistent sign relative to the Z-axis, ensure the normal points generally "up"
    if (target_normal.z() < 0.0f) {
        target_normal = -target_normal;
    }

    float deviation_sum = 0.0f;
    Eigen::Vector3f point_vec(_point.x, _point.y, _point.z);
    for (const auto& neighbor_point : _cloud->points) {
        Eigen::Vector3f neighbor_vec(neighbor_point.x, neighbor_point.y, neighbor_point.z);
        // Vector from our point to its neighbor
        Eigen::Vector3f diff = neighbor_vec - point_vec;
        deviation_sum += diff.dot(target_normal);
    }

    float sign = (deviation_sum > 0.0f) ? 1.0f : -1.0f;
    
    // Step 4: Compute principal curvatures for the whole cloud
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> ce;
    ce.setInputCloud(curvatureCloud);
    ce.setInputNormals(normals);
    ce.setRadiusSearch(_radius);

    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    ce.compute(*curvatures);

    // Step 6: Print out results
    const auto& curvature = curvatures->points[target_idx];
    float pc1 = sign*curvature.pc1;
    float pc2 = sign*curvature.pc2;
    float mean_curvature = sign * (pc1 + pc2) / 2.0f;
    float gaussian_curvature = pc1 * pc2;

    std::cout << "Point " << target_idx << ": Principal Curvatures: " << pc1 << ", " << pc2
                << ", Mean Curvature: " << mean_curvature
                << ", Gaussian Curvature: " << gaussian_curvature << std::endl;
    std::cout << "  Principal Directions: (" << curvature.principal_curvature_x << ", "
                << curvature.principal_curvature_y << ", " << curvature.principal_curvature_z << ")" << std::endl;

    if (gaussian_curvature < -1e-5) { // Use a small threshold for floating point errors
        std::cout << "  Surface Type: Saddle Point" << std::endl;
    } else if (gaussian_curvature > 1e-5) {
        std::cout << "  Surface Type: " << (sign > 0 ? "Bowl (Concave)" : "Dome (Convex)") << std::endl;
    } else {
        std::cout << "  Surface Type: Plane or Cylinder" << std::endl;
    }

    // Step 6: Return the curvature of the target point (last point in the cloud)
    if (!curvatures->empty()) {
        return curvatures->points[target_idx]; // The last point corresponds to the added point
    } else {
        std::cout << "Error: No curvature computed for the specified point." << std::endl;
        // throw std::runtime_error("No curvature computed for the specified point.");
        return pcl::PrincipalCurvatures();
    }
}

pcl::ModelCoefficients::Ptr computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud)
{
    if (_cloud->empty()) {
        std::cerr << "Error: Input cloud for computePlane cannot be empty." << std::endl;
        // Return a null pointer to indicate failure, which is standard for pointer return types.
        return nullptr;
    }

    // --- The plane finding logic is the same ---

    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_cloud, centroid);

    // Perform PCA
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(_cloud);
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

    // The eigenvector corresponding to the smallest eigenvalue is the normal of the plane
    Eigen::Vector3f normal = eigenvectors.col(2);

    // Compute the plane coefficients (Ax + By + Cz + D = 0)
    float a = normal[0];
    float b = normal[1];
    float c = normal[2];
    float d = -(normal.dot(centroid.head<3>()));

    // --- The return type creation is new ---

    // Create and populate the ModelCoefficients object
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    coefficients->values.resize(4);
    coefficients->values[0] = a;
    coefficients->values[1] = b;
    coefficients->values[2] = c;
    coefficients->values[3] = d;
    
    // Print the plane coefficients for verification
    std::cout << "Plane coefficients (Ax + By + Cz + D = 0):\n";
    std::cout << "a: " << coefficients->values[0] << "\n";
    std::cout << "b: " << coefficients->values[1] << "\n";
    std::cout << "c: " << coefficients->values[2] << "\n";
    std::cout << "d: " << coefficients->values[3] << "\n";

    return coefficients;
}

float computePlaneAngle(const pcl::ModelCoefficients::Ptr _coefficients)
{
    // Extract the normal vector (nx, ny, nz)
    Eigen::Vector3f normal(_coefficients->values[0], _coefficients->values[1], _coefficients->values[2]);

    // Normalize the normal vector (in case it's not already normalized)
    normal.normalize();

    // Define the Z-axis vector
    Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);

    // Compute the dot product between the normal and the Z-axis
    float dot_product = normal.dot(z_axis);

    // Compute the angle in radians
    double theta = std::acos(dot_product);

    // Convert to degrees
    double theta_deg = theta * 180.0 / M_PI;

    double slope = theta_deg;
    if (theta_deg > 90.0) {
        slope = 180.0 - theta_deg;
    }

    // Print the results
    std::cout << "Angle of slope: " << slope << " degrees\n";

    return slope;
}

float pointToPlaneDistance(const pcl::PointXYZRGB& _point, const pcl::ModelCoefficients::Ptr _coefficients)
{
    float a = _coefficients->values[0];
    float b = _coefficients->values[1];
    float c = _coefficients->values[2];
    float d = _coefficients->values[3];

    // Compute the distance
    float distance = std::abs(a * _point.x + b * _point.y + c * _point.z + d);
    return distance;
}

float computeStandardDeviation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const pcl::ModelCoefficients::Ptr _coefficients)
{
    std::vector<float> distances;
    for (const auto& point : _cloud->points) {
        float distance = pointToPlaneDistance(point, _coefficients);
        distances.push_back(distance);
    }

    // Compute the mean distance
    float mean = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

    // Compute the variance
    float variance = 0.0f;
    for (float distance : distances) {
        variance += std::pow(distance - mean, 2);
    }
    variance /= distances.size();

    // Compute the standard deviation
    float std_dev = std::sqrt(variance);
    std::cout << "Standard Deviation: " << std_dev << std::endl;
    return std_dev;
}

pcl::PointIndices findLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _radius,
    const bool _isMin)
{
    pcl::Indices indices;
    pcl::PointCloud<pcl::PointXYZRGB> inputCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));

    if(_isMin)
    {
        for(auto& point : *cloud_copy)
        {
            point.z = -point.z;
        }
    }

    pcl::LocalMaximum< pcl::PointXYZRGB > lm;
    lm.setNegative(true);
    lm.setRadius(_radius);
    lm.setInputCloud(cloud_copy);
    lm.filter(indices);

    pcl::PointIndices pointIndices;
    pointIndices.indices = indices;

    return pointIndices;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractLocalExtremums(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const float _radius,
    const bool _isMin)
{
    pcl::PointIndices idx = findLocalExtremums(_cloud, _radius, _isMin);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr localExtremums(new pcl::PointCloud<pcl::PointXYZRGB>);
    localExtremums = extractPoints<pcl::PointXYZRGB>(_cloud, idx, false);
    return localExtremums;
}

float computePointsDist2D(
    const pcl::PointXYZRGB& _point1,
    const pcl::PointXYZRGB& _point2)
{
    float dx = _point1.x - _point2.x;
    float dy = _point1.y - _point2.y;
    return std::sqrt(dx * dx + dy * dy);
}

float computePointsDist3D(
    const pcl::PointXYZRGB& _point1,
    const pcl::PointXYZRGB& _point2)
{
    float dx = _point1.x - _point2.x;
    float dy = _point1.y - _point2.y;
    float dz = _point1.z - _point2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// float computeMinDistToBB(
//     const pcl::PointXYZRGB& _landingPoint,
//     const BoundingBox& _treeBB) {
//     float min_x_dist = std::min(_landingPoint.x - _treeBB.min_x, _treeBB.max_x - _landingPoint.x);
//     float min_y_dist = std::min(_landingPoint.y - _treeBB.min_y, _treeBB.max_y - _landingPoint.y);
//     float min_dist = std::min(min_x_dist, min_y_dist);

//     return min_dist;
// }

DistsOfInterest computeDistToPointsOfInterest(
    const pcl::PointXYZRGB& _landingPoint,
    const std::vector<pcl::PointXYZRGB>& _pointsOfInterest,
    const pcl_tools::OrientedBoundingBox& _treeBB)
{
    if(_pointsOfInterest.size() != 3){
        std::cout << "_pointsOfInterest != 3 in computeDistToPointsOfInterest()" << std::endl;
        return DistsOfInterest();
    }

    std::vector<std::vector<float>> vec_distsOfInterest;
    vec_distsOfInterest.reserve(_pointsOfInterest.size());

    float min_diameter = std::min(_treeBB.width, _treeBB.height);
    float min_radius = min_diameter / 2.0;
    for(size_t i = 0; i < _pointsOfInterest.size(); ++i)
    {
        const pcl::PointXYZRGB pointOfInterest = _pointsOfInterest[i];

        float dist2D = computePointsDist2D(_landingPoint, pointOfInterest);
        float dist3D = computePointsDist3D(_landingPoint, pointOfInterest);

        float ratio2D = dist2D/min_radius;
        float ratio3D = dist3D/min_radius;

        std::vector<float> row;
        row.push_back(dist2D);
        row.push_back(dist3D);
        row.push_back(ratio2D);
        row.push_back(ratio3D);
        vec_distsOfInterest.emplace_back(row);

        std::cout << "Distance to Point of interest #" << i << " 2D: " << dist2D << std::endl;
        std::cout << "Distance to Point of interest #" << i << " 3D: " << dist3D << std::endl;
        std::cout << "Ratio Distance 2D to Point of interest over tree diameter #" << i << ": " << ratio2D << std::endl;
        std::cout << "Ratio Distance 3D to Point of interest over tree diameter #" << i << ": " << ratio3D << std::endl;
    }

    DistsOfInterest distsOfInterest;

    distsOfInterest.distTreeCenter2D = vec_distsOfInterest[0][0];
    distsOfInterest.distTreeCenter3D = vec_distsOfInterest[0][1];
    distsOfInterest.ratioTreeCenter2D = vec_distsOfInterest[0][2];
    distsOfInterest.ratioTreeCenter3D = vec_distsOfInterest[0][3];

    distsOfInterest.distTreeHighestPoint2D = vec_distsOfInterest[1][0];
    distsOfInterest.distTreeHighestPoint3D = vec_distsOfInterest[1][1];
    distsOfInterest.ratioTreeHighestPoint2D = vec_distsOfInterest[1][2];
    distsOfInterest.ratioTreeHighestPoint3D = vec_distsOfInterest[1][3];

    distsOfInterest.distTreeMidwayPoint2D = vec_distsOfInterest[2][0];
    distsOfInterest.distTreeMidwayPoint3D = vec_distsOfInterest[2][1];
    distsOfInterest.ratioTreeMidwayPoint2D = vec_distsOfInterest[2][2];
    distsOfInterest.ratioTreeMidwayPoint3D = vec_distsOfInterest[2][3];

    return distsOfInterest;
}

// bool isPointIn2DPolygon(const pcl::PointXYZRGB& point, const pcl::PointCloud<pcl::PointXYZRGB>& polygon)
// {
//     bool is_inside = false;
//     int n_points = polygon.points.size();

//     // Iterate over each edge of the polygon (from point i to point j)
//     for (int i = 0, j = n_points - 1; i < n_points; j = i++)
//     {
//         const auto& p_i = polygon.points[i];
//         const auto& p_j = polygon.points[j];

//         // Check if the point's y-coordinate is between the edge's y-coordinates.
//         // Also checks if the point's x-coordinate is to the left of the edge's x-coordinates.
//         if (((p_i.y > point.y) != (p_j.y > point.y)) &&
//             (point.x < (p_j.x - p_i.x) * (point.y - p_i.y) / (p_j.y - p_i.y) + p_i.x))
//         {
//             // If the conditions are met, the ray from the point intersects this edge.
//             // We flip the 'is_inside' boolean. An odd number of intersections means
//             // the point is inside; an even number means it's outside.
//             is_inside = !is_inside;
//         }
//     }
//     return is_inside;
// }

// Helper struct to pair a point with its angle for sorting
struct PointWithAngle {
    pcl::PointXYZRGB point;
    float angle;
};

// Custom comparison function for std::sort
bool comparePointsByAngle(const PointWithAngle& a, const PointWithAngle& b) {
    return a.angle < b.angle;
}

/**
 * @brief Sorts the vertices of a 2D polygon (point cloud) into a counter-clockwise order.
 * @param _polygon The point cloud representing the polygon vertices. This cloud is modified in place.
 */
void sortPolygonVertices(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _polygon) {
    if (_polygon->size() < 3) {
        // Not enough points to form a polygon, do nothing.
        return;
    }

    // 1. Calculate the 2D centroid of the polygon.
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_polygon, centroid);

    // 2. Create a vector of points with their angles relative to the centroid.
    std::vector<PointWithAngle> points_with_angles;
    points_with_angles.reserve(_polygon->size());
    for (const auto& pt : _polygon->points) {
        float angle = std::atan2(pt.y - centroid[1], pt.x - centroid[0]);
        points_with_angles.push_back({pt, angle});
    }

    // 3. Sort the points based on their calculated angle.
    std::sort(points_with_angles.begin(), points_with_angles.end(), comparePointsByAngle);

    // 4. Recreate the point cloud with the sorted points.
    _polygon->points.clear();
    for (const auto& pwa : points_with_angles) {
        _polygon->points.push_back(pwa.point);
    }
    
    // Update width/height for consistency.
    _polygon->width = _polygon->points.size();
    _polygon->height = 1;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr computeConcaveHull2D(
    const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& _cloud,
    double _alpha)
{
    // --- Input Validation ---
    if (_cloud->points.size() < 3) {
        std::cerr << "Error: Input cloud needs at least 3 points to form a polygon." << std::endl;
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr();
    }

    // --- 1. Create Hull ---
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_polygon(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    pcl::ConcaveHull<pcl::PointXYZRGB> chull;
    chull.setInputCloud(_cloud);
    chull.setAlpha(_alpha);
    chull.setDimension(2);
    chull.reconstruct(*hull_polygon);

    sortPolygonVertices(hull_polygon);

    if (hull_polygon->points.size() < 3) {
        std::cerr << "Error: Concave hull resulted in fewer than 3 points. Try a larger alpha value." << std::endl;
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr();
    }

    std::cout << "_cloud size: " << _cloud->points.size() << std::endl;
    std::cout << "hull_polygon size: " << hull_polygon->points.size() << std::endl;

    return hull_polygon;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr  extractConcaveHullArchive(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl_tools::OrientedBoundingBox& _treeBB,
    const int _n_neighbors_search,
    const double _alpha)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr treeNormMatchedCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_treeCloud));
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud = pcl_tools::extractNormalsPC(
        treeNormMatchedCloud,
        _n_neighbors_search,
        pcl::PointXYZRGB(_treeBB.centroid[0], _treeBB.centroid[1], _treeBB.centroid[2], 255, 255, 255)
    );
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr boundaryCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*treeNormMatchedCloud));
    pcl_tools::extractBoundary(boundaryCloud, normalsCloud, _n_neighbors_search);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr hull_polygon = pcl_tools::computeConcaveHull2D(
        boundaryCloud,
        _alpha
    );

    return hull_polygon;
}

Features computeFeatures(
    const pcl::PointXYZRGB& _landingPoint,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _segCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl_tools::OrientedBoundingBox& _treeBB,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _hull_polygon,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _landingSurfaceCloud,
    const float& _lz_factor,
    const float& _radius)
{
    pcl::PointXYZRGB treeCenterPoint(_treeBB.centroid[0], _treeBB.centroid[1], 0.0, 255, 255, 255);
    projectPoint(_segCloud, treeCenterPoint);

    pcl::PrincipalCurvatures curvatures = computeCurvature(_landingSurfaceCloud, _landingPoint, _lz_factor*_radius);
    float density = computeSurfaceDensity(_landingSurfaceCloud, _lz_factor*_radius);
    pcl::ModelCoefficients::Ptr coef = computePlane(_landingSurfaceCloud);
    float slope = computePlaneAngle(coef);
    float stdDev = computeStandardDeviation(_landingSurfaceCloud, coef);

    pcl::PointXYZRGB highestPoint = getHighestPoint(_treeCloud);

    pcl::PointXYZRGB midwayPoint;
    midwayPoint.x = (treeCenterPoint.x + highestPoint.x) / 2.0f;
    midwayPoint.y = (treeCenterPoint.y + highestPoint.y) / 2.0f;
    projectPoint(_segCloud, midwayPoint);

    DistsOfInterest distsOfInterest = computeDistToPointsOfInterest(
        _landingPoint, 
        std::vector<pcl::PointXYZRGB>({
            treeCenterPoint,
            highestPoint,
            midwayPoint
        }),
        _treeBB
    );

    distsOfInterest.distTop = highestPoint.z - _landingPoint.z;
    distsOfInterest.distBbox2D = distanceToOBB2D(_landingPoint, _treeBB);
    double min_tree_dia = std::min(_treeBB.width, _treeBB.height);
    double min_tree_radius = min_tree_dia / 2.0;
    distsOfInterest.ratioBbox2D = distsOfInterest.distBbox2D / min_tree_radius;
    distsOfInterest = computeDistancesToPolygon(_landingPoint, _hull_polygon, _treeBB, distsOfInterest);

    return Features{curvatures, _treeBB, density, slope, stdDev, distsOfInterest, coef};
}

// Features computeLandingPointFeatures(
//     const pcl::PointXYZRGB& _landingPoint,
//     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud)
// {
//     OrientedBoundingBox treeBB = getOBB(_treeCloud);

//     pcl::PointXYZRGB treeCenterPoint(treeBB.centroid[0], treeBB.centroid[1], 0.0, 255, 255, 255);
//     projectPoint(_treeCloud, treeCenterPoint);

//     pcl::PointXYZRGB highestPoint = getHighestPoint(_treeCloud);

//     DistsOfInterest distsOfInterest = computeDistToPointsOfInterest(
//         _landingPoint, 
//         std::vector<pcl::PointXYZRGB>({
//             treeCenterPoint,
//             highestPoint
//         }),
//         treeBB
//     );

//     distsOfInterest.distTop = highestPoint.z - _landingPoint.z;

//     return Features{
//         pcl::PrincipalCurvatures(),
//         treeBB,
//         std::numeric_limits<float>::quiet_NaN(),
//         std::numeric_limits<float>::quiet_NaN(),
//         std::numeric_limits<float>::quiet_NaN(),
//         distsOfInterest,
//         pcl::ModelCoefficients::Ptr()
//     };
// }

// Features computeLandingZoneFeatures(
//     const pcl::PointXYZRGB& _landingPoint,
//     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
//     const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _landingSurfaceCloud,
//     const float& _lz_factor,
//     const float& _radius,
//     const Features& _landing_point_features)
// {
//     pcl::PointXYZRGB treeCenterPoint(_landing_point_features.treeBB.centroid[0], _landing_point_features.treeBB.centroid[1], 0.0, 255, 255, 255);
//     projectPoint(_treeCloud, treeCenterPoint);

//     pcl::PrincipalCurvatures curvatures = computeCurvature(_landingSurfaceCloud, _landingPoint, _lz_factor*_radius);
//     float density = computeSurfaceDensity(_landingSurfaceCloud, _lz_factor*_radius);
//     pcl::ModelCoefficients::Ptr coef = computePlane(_landingSurfaceCloud);
//     float slope = computePlaneAngle(coef);
//     float stdDev = computeStandardDeviation(_landingSurfaceCloud, coef);

//     return Features{curvatures, _landing_point_features.treeBB, density, slope, stdDev, _landing_point_features.distsOfInterest, coef};
// }

std::vector<pcl_tools::Features> computeFeaturesList(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _segCloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _treeCloud,
    const pcl_tools::OrientedBoundingBox& _treeBB,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _hull_polygon,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _gridCloud,
    const float& _landing_zone_factor,
    const float& _radius,
    const float& _min_lz_points)
{
    std::vector<Features> features_list;
    for(const auto& point : _gridCloud->points) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr landingSurfaceCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_segCloud));
        extractNeighborCirclePC(landingSurfaceCloud, point, _landing_zone_factor*_radius);

        if(landingSurfaceCloud->size() < _min_lz_points) {
            features_list.push_back(Features());
            continue;
        }

        Features features = computeFeatures(point, _segCloud, _treeCloud, _treeBB, _hull_polygon, landingSurfaceCloud, _landing_zone_factor, _radius);
        features_list.push_back(features);
    }

    return features_list;
}

void printPoint(const pcl::PointXYZRGB& _point)
{
    std::cout << "Point: ("
    << _point.x << ", "
    << _point.y << ", "
    << _point.z << ")" << std::endl;
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _coloredCloud,
    const pcl::RGB& _color)
{
    for (auto& point : _coloredCloud->points)
    {
        point.r = _color.r;
        point.g = _color.g;
        point.b = _color.b;
    }
}

void colorSegmentedPoints(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _coloredCloud,
    const pcl::PointIndices& inliers,
    const pcl::RGB& _color)
{
    // Assign the same color to all points in the segmented region
    for (std::size_t i = 0; i < inliers.indices.size(); ++i)
    {
        const auto& idx = inliers.indices[i];

        if(_coloredCloud->points[idx].r == _coloredCloud->points[idx].g && _coloredCloud->points[idx].g == _coloredCloud->points[idx].b)
        {
            _coloredCloud->points[idx].r = _color.r;
            _coloredCloud->points[idx].g = _color.g;
            _coloredCloud->points[idx].b = _color.b;
        }
        else
        {
            if(_coloredCloud->points[idx].r == 0){_coloredCloud->points[idx].r = _color.r;}
            if(_coloredCloud->points[idx].g == 0){_coloredCloud->points[idx].g = _color.g;}
            if(_coloredCloud->points[idx].b == 0){_coloredCloud->points[idx].b = _color.b;}
        }
    }
}

bool checkInboundPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _ogCloud, const pcl::PointXYZRGB& _landing)
{
    pcl::PointXYZRGB min_pt, max_pt;
    pcl::getMinMax3D(*_ogCloud, min_pt, max_pt);

    bool isPointInBound = true;
    // Bounding box dimensions
    float width = max_pt.x - min_pt.x;
    float height = max_pt.y - min_pt.y;
    float depth = max_pt.z - min_pt.z;
    float c_x = width / 2.0 + min_pt.x;
    float c_y = height / 2.0 + min_pt.y;

    // std::cout << "\n" << "AABB Dimensions: "
    //         << width << " (W) x "
    //         << height << " (H) x "
    //         << depth << " (D)\n\n";

    if(_landing.x < min_pt.x || _landing.x > max_pt.x){
        std::cout << "\n\nWARNING: Target point out of bound in x: " << min_pt.x << " < " << _landing.x << " < " << max_pt.x << "\n\n";
        isPointInBound = false;
    }
    if(_landing.y < min_pt.y || _landing.y > max_pt.y){
        std::cout << "\n\nWARNING: Target point out of bound in y: " << min_pt.y << " < " << _landing.y << " < " << max_pt.y << "\n\n";
        isPointInBound = false;
    }

    if(_landing.z < min_pt.z || _landing.z > max_pt.z){
        std::cout << "\n\nWARNING: Target point out of bound in z: " << min_pt.z << " < " << _landing.z << " < " << max_pt.z << "\n\n";
        isPointInBound = false;
    }

    return isPointInBound;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centerItems4Viewing(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> _clouds,
    OrientedBoundingBox* _obb = nullptr,
    pcl::PointXYZRGB* _model = nullptr,
    std::vector<pcl::PointXYZRGB>* _spheres = nullptr,
    pcl::ModelCoefficients::Ptr _plane = nullptr)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centered_clouds;
    if (_clouds.empty() || !_clouds[0]) {
        return centered_clouds;
    }

    // Define the transform once, outside the if/else
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();

    if(_obb != nullptr) {
        // 1. Get transform from OBB centroid
        transform.translation() << -_obb->centroid[0], -_obb->centroid[1], -_obb->centroid[2];
        _obb->centroid = Eigen::Vector3f::Zero();
        std::cout << "Centered OBbox" << std::endl;
    }
    else {
        // 1. Get transform from the first cloud's centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*_clouds[0], centroid);
        transform.translation() << -centroid[0], -centroid[1], -centroid[2];
    }

    // 2. Apply the transform to all clouds
    for (const auto& cloud : _clouds) {
        auto centered_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        if (cloud) {
            pcl::transformPointCloud(*cloud, *centered_cloud, transform);
        }
        centered_clouds.push_back(centered_cloud);
    }
    std::cout << "Centered clouds" << std::endl;

    if(_model != nullptr) {
        Eigen::Vector3f point_as_eigen = _model->getVector3fMap();
        Eigen::Vector3f transformed_point = transform * point_as_eigen;
        _model->x = transformed_point.x();
        _model->y = transformed_point.y();
        _model->z = transformed_point.z();
        std::cout << "Centered model" << std::endl;
    }

    // 3. Apply the transform to all spheres (if they exist)
    if(_spheres != nullptr) {
        for(auto& sphere : *_spheres) {
            Eigen::Vector3f point_as_eigen = sphere.getVector3fMap();
            Eigen::Vector3f transformed_point = transform * point_as_eigen;
            sphere.x = transformed_point.x();
            sphere.y = transformed_point.y();
            sphere.z = transformed_point.z();
        }
        std::cout << "Centered spheres" << std::endl;
    }

    // 4. Apply the transform to the plane (if it exists)
    if(_plane != nullptr && !_plane->values.empty()) {
        Eigen::Vector4f input_plane_coeffs(_plane->values.data());
        Eigen::Vector4f transformed_plane_coeffs;
        pcl::transformPlane(input_plane_coeffs, transformed_plane_coeffs, transform);
        Eigen::Map<Eigen::Vector4f>(_plane->values.data()) = transformed_plane_coeffs;
        std::cout << "Centered plane" << std::endl;
    }

    return centered_clouds;
}

void addCloud2View(pcl::visualization::PCLVisualizer::Ptr _viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const std::string& _name)
{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    _viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, _name);
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _name);
}

void view(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& _clouds,
    const OrientedBoundingBox* _obb,
    const pcl::PointXYZRGB* _model,
    const std::vector<pcl::PointXYZRGB>* _spheres,
    const pcl::ModelCoefficients::Ptr _plane)
{
    std::cout << "Entering view" << std::endl;

    OrientedBoundingBox view_obb;
    OrientedBoundingBox* p_view_obb = nullptr; // Pointer to our local copy
    if (_obb != nullptr) {
        view_obb = *_obb;
        p_view_obb = &view_obb;
    }

    pcl::PointXYZRGB view_model;
    pcl::PointXYZRGB* p_view_model = nullptr; // Pointer to our local copy
    if (_model != nullptr) {
        view_model = *_model;
        p_view_model = &view_model;
    }

    // Copy spheres if they exist
    std::vector<pcl::PointXYZRGB> view_spheres;
    std::vector<pcl::PointXYZRGB>* p_view_spheres = nullptr; // Pointer to our local copy
    if (_spheres != nullptr) {
        view_spheres = *_spheres;
        p_view_spheres = &view_spheres;
    }

    // Copy plane if it exists
    pcl::ModelCoefficients::Ptr view_plane = nullptr; // This is a shared_ptr
    if (_plane != nullptr) {
        // Create a deep copy of the ModelCoefficients object
        view_plane = pcl::make_shared<pcl::ModelCoefficients>(*_plane);
    }

    // 2. Make ONE call to centerItems4Viewing.
    //    This call uses our local, modifiable pointers (p_view_obb, etc.).
    //    The function will modify view_obb, view_spheres, and view_plane in place.
    std::cout << "Calling centerItems4Viewing for clouds" << std::endl;
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centered_clouds = 
        centerItems4Viewing(_clouds, p_view_obb, p_view_model, p_view_spheres, view_plane);
    std::cout << "Called centerItems4Viewing for clouds" << std::endl;
    
    std::cout << "Centered all items" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(1.0, 1.0, 1.0);

    if(_model != nullptr) {
        // --- 1. USE THE ASCII FILE ---
        std::string ply_filename = "/home/docker/tree_landing_eda/data/inputs/DRONE_WITH_PETALS.ply";
        std::string model_id = "my_model";

        // --- 2. Create a PCL PolygonMesh object ---
        pcl::PolygonMesh::Ptr model_mesh(new pcl::PolygonMesh);

        // --- 3. Load the file using PCL's native PLY mesh loader ---
        if (pcl::io::loadPLYFile(ply_filename, *model_mesh) == -1) {
            std::cerr << "!!! ERROR: FAILED to load PLY as a PolygonMesh from " << ply_filename << std::endl;
        
        } else {
            std::cout << "--- Successfully loaded PLY as a PolygonMesh ---" << std::endl;
            std::cout << "Mesh has " << model_mesh->cloud.width * model_mesh->cloud.height << " vertices." << std::endl;
            std::cout << "Mesh has " << model_mesh->polygons.size() << " faces." << std::endl;

            // --- 1. Create Scaling Transform ---
            Eigen::Affine3f T_scale = Eigen::Affine3f::Identity();
            // Scale by 0.001 to convert mm (mesh) to meters (clouds)
            T_scale.scale(1.0); 

            // --- 2. Create Centered Translation Transform ---
            // Use 'view_model', which holds the CENTERED position
            Eigen::Affine3f T_model_centered = Eigen::Affine3f::Identity();
            T_model_centered.translation() << view_model.x, view_model.y, view_model.z;

            // --- 3. Combine them: Scale first, then Translate ---
            Eigen::Affine3f transform_combined = T_model_centered * T_scale;

            // --- 4. Apply the transform ---
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::fromPCLPointCloud2(model_mesh->cloud, *temp_cloud);
            pcl::transformPointCloud(*temp_cloud, *temp_cloud, transform_combined);
            pcl::toPCLPointCloud2(*temp_cloud, model_mesh->cloud);

            // --- 4. Add the PCL mesh object to the viewer ---
            if (viewer->addPolygonMesh(*model_mesh, model_id))
            {           
                // --- 5. Set properties ONLY if mesh was added successfully ---
                std::cout << "--- Successfully added mesh to viewer with ID: " << model_id << " ---" << std::endl;
                // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.5, 0.5, 0.5, model_id);
                // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.7, model_id);
            }
            else
            {
                // --- 5. Failed to add mesh ---
                std::cerr << "!!! ERROR: FAILED to add PolygonMesh to viewer with ID: " << model_id << std::endl;
            }
        }
    }

    int i = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud : centered_clouds) {
        std::string cloud_id = "cloud" + std::to_string(i);
        addCloud2View(viewer, cloud, cloud_id);
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 
            5, 
            cloud_id
        );
        ++i;
    }
    if(_obb != nullptr) {
        viewer->addCube(view_obb.centroid, view_obb.rotation, view_obb.width, view_obb.height, view_obb.depth, "oriented_bbox");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "oriented_bbox");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "oriented_bbox"); // Red
    }
    if(_spheres != nullptr) {
        double radius = 0.2;
        int sphere_id = 0;
        for(const auto& sphere : view_spheres) {
            std::string unique_id = "centroid_" + std::to_string(sphere_id);
            viewer->addSphere(sphere, radius, sphere.r / 255.0, sphere.g / 255.0, sphere.b / 255.0, unique_id);
            ++sphere_id;
        }
    }
    if(_plane != nullptr){
        viewer->addPlane(*view_plane, "plane");
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 0, "plane");
    }

    viewer->addCoordinateSystem(1.0);
    // viewer->initCameraParameters();

    viewer->setCameraPosition(
        -12.0, -12.0, 25.0,   // Camera position
        0.0, 0.0, 0.0,   // Viewpoint (looking at the origin)
        0.0, 0.0, 1.0    // Up vector (Z-axis is up)
    );

    // https://github.com/PointCloudLibrary/pcl/issues/5237#issuecomment-1114255056
    // spin() instead of spinOnce() avoids crash
    viewer->spin();
}

}
