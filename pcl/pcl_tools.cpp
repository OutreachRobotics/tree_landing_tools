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

std::vector<pcl::PointIndices> mapPointsToSegments(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud, // The original, full point cloud
    const cv::Mat& _markers,                               // The segmentation result
    const DepthMapData& _mapData)                          // Contains min_x, min_y
{
    // A map to hold the clusters, keyed by their segment label
    std::map<int, pcl::PointIndices::Ptr> temp_segment_map;

    // 1. Iterate through EVERY point in the original cloud
    for (int i = 0; i < _cloud->points.size(); ++i) {
        const auto& point = _cloud->points[i];

        // 2. Project the 3D point back to 2D grid/pixel coordinates
        int grid_x = static_cast<int>(floor(point.x / _mapData.leafSize));
        int grid_y = static_cast<int>(floor(point.y / _mapData.leafSize));

        int u = grid_x - _mapData.min_x; // Pixel column
        int v = grid_y - _mapData.min_y; // Pixel row

        // 3. Check if the point's projection falls within the bounds of the marker image
        if (v >= 0 && v < _markers.rows && u >= 0 && u < _markers.cols) {
            // 4. Get the segment label from the corresponding pixel in the marker image
            int label = _markers.at<int>(v, u);

            // Skip background (label 1) and watershed boundaries (label -1)
            if (label <= 1) {
                continue;
            }

            // If this is the first point for a new segment, create an entry in the map
            if (temp_segment_map.find(label) == temp_segment_map.end()) {
                temp_segment_map[label] = pcl::make_shared<pcl::PointIndices>();
            }

            // 5. Add the original point's index to the correct cluster
            temp_segment_map[label]->indices.push_back(i);
        }
    }

    // Convert the map to the final output vector
    std::vector<pcl::PointIndices> final_clusters;
    for (const auto& pair : temp_segment_map) {
        final_clusters.push_back(*pair.second);
    }

    return final_clusters;
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

    // Loop through all pairs of blobs to find Pac-Man configurations.
    for (int eater_idx = 1; eater_idx < num_labels; ++eater_idx) {
        std::cout << "potential eater solidity: " << solidity[eater_idx] << std::endl;
        if (solidity[eater_idx] < _solidity_threshold) {
            std::cout << "eater solidity: " << solidity[eater_idx] << std::endl;

            for (int eaten_idx = 1; eaten_idx < num_labels; ++eaten_idx) {
                if (eater_idx == eaten_idx) continue;

                // cv::Point2d eaten_centroid(centroids.at<double>(eaten_idx, 0), centroids.at<double>(eaten_idx, 1));
                // if (!hulls[eater_idx].empty() && !contours[eaten_idx].empty()) {
                //     int inside_points_count = 0;
                //     // Loop through each point of the "eaten" blob's contour.
                //     for (const cv::Point& pt : contours[eaten_idx]) {
                //         if (cv::pointPolygonTest(hulls[eater_idx], pt, false) >= 0) {
                //             inside_points_count++;
                //         }
                //     }

                //     // Check if a significant percentage of points are inside.
                //     double percentage_inside = static_cast<double>(inside_points_count) / contours[eaten_idx].size();
                //     std::cout << "potential eaten percentage: " << percentage_inside << std::endl;
                //     if (percentage_inside > _percentage_threshold) {
                //         std::cout << "eaten percentage: " << percentage_inside << std::endl;

                //         int eater_orig_label = _markers.at<int>(safe_points[eater_idx]);
                //         int eaten_orig_label = _markers.at<int>(safe_points[eaten_idx]);

                //         std::cout << "Testing blob " << eaten_orig_label << " into blob " << eater_orig_label << std::endl;
                //         if (eater_orig_label > 1 && eaten_orig_label > 1 && eater_orig_label != eaten_orig_label) {
                //             std::cout << "Fusing blob " << eaten_orig_label << " into blob " << eater_orig_label << std::endl;
                //             remap_table[eaten_orig_label] = eater_orig_label;
                //         }
                //     }
                // }

                // Get the centroid of the potential "eater" blob.
                cv::Point2d eater_centroid(centroids.at<double>(eater_idx, 0), centroids.at<double>(eater_idx, 1));
                cv::Point2d eaten_centroid(centroids.at<double>(eaten_idx, 0), centroids.at<double>(eaten_idx, 1));
                std::cout << "eater centroid: " << eater_centroid << std::endl;
                std::cout << "eaten centroid: " << eaten_centroid << std::endl;
                
                // Check if the eaten's hull is valid.
                if (!hulls[eater_idx].empty() && !hulls[eaten_idx].empty()) {
                    // Test if the eaten blob's centroid is inside or on the eater's convex hull.
                    double distance_nr = cv::pointPolygonTest(hulls[eaten_idx], eater_centroid, true);
                    double distance_rn = cv::pointPolygonTest(hulls[eater_idx], eaten_centroid, true);
                    std::cout << "Distance from eater centroid to eaten hull: " << distance_nr << std::endl;
                    std::cout << "Distance from eaten centroid to eater hull: " << distance_rn << std::endl;
                    if (distance_nr >= -1.0 || distance_rn >= -1.0) {
                        int eater_orig_label = _markers.at<int>(safe_points[eater_idx]);
                        int eaten_orig_label = _markers.at<int>(safe_points[eaten_idx]);

                        if (eater_orig_label > 1 && eaten_orig_label > 1 && eater_orig_label != eaten_orig_label) {
                            std::cout << "Fusing blob " << eaten_orig_label << " into blob " << eater_orig_label << std::endl;
                            remap_table[eaten_orig_label] = eater_orig_label;
                        }
                    }
                }
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
    std::set<int> unique_labels;
    for (int y = 0; y < _markers.rows; ++y) {
        for (int x = 0; x < _markers.cols; ++x) {
            int& label = _markers.at<int>(y, x);
            if (remap_table.count(label)) {
                label = remap_table[label];
            }
            if (label > 1) { // Assuming 0 is unknown and 1 is background
                unique_labels.insert(label);
            }
        }
    }

    // A small kernel is enough to bridge a 1-pixel gap.
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // For each unique blob, apply closing to fill its internal gaps.
    for (int label : unique_labels) {
        // Create a mask for the current blob only.
        cv::Mat blob_mask = (_markers == label);

        // Close the gaps in the mask.
        cv::morphologyEx(blob_mask, blob_mask, cv::MORPH_CLOSE, kernel);

        // Update the main markers image with the filled blob.
        _markers.setTo(label, blob_mask);
    }
}

std::vector<pcl::PointIndices> segmentWatershed(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& _cloud,
    const float _leafSize,
    const float _radius,
    const int _medianKernelSize,
    const int _tophat_kernel,
    const float _tophat_amplification,
    const float _pacman_solidity,
    const bool _shouldView)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr smoothCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    smoothPC(smoothCloud, 2.0*_radius);

    DepthMapData depthMapData = computeDepthMap(_cloud, _leafSize, _medianKernelSize);
    DepthMapData smoothDepthMapData = computeDepthMap(smoothCloud, _leafSize, _medianKernelSize);

    // Define "sure background" as all pixels with 0 value in the original depth map
    cv::Mat sure_bg;
    cv::compare(smoothDepthMapData.depthMap, 0, sure_bg, cv::CMP_EQ);

    // Create an inverted version of the background mask
    cv::Mat foreground_mask;
    cv::bitwise_not(sure_bg, foreground_mask);

    // --- Method 2: Get seeds from 3D Local Extrema ---
    pcl::PointIndices maximumIdx = pcl_tools::findLocalExtremums(smoothCloud, _radius, false);
    cv::Mat sure_fg_extremums = cv::Mat::zeros(smoothDepthMapData.depthMap.size(), CV_8UC1);

    for (const int& index : maximumIdx.indices) {
        const auto& point = smoothCloud->points[index];

        // Project the 3D point back to 2D grid/pixel coordinates.
        int grid_x = static_cast<int>(floor(point.x / _leafSize));
        int grid_y = static_cast<int>(floor(point.y / _leafSize));

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
    fusePacManBlobs(markers, _pacman_solidity);

    std::vector<pcl::PointIndices> clusters = mapPointsToSegments(_cloud, markers, smoothDepthMapData);
    // extractBiggestSegment(_cloud, clusters);

    if(_shouldView){
        // 5. VISUALIZE THE RESULT
        cv::Mat dist_markers_viz = visualizeMarkers(dist_markers);
        cv::Mat markers_viz = visualizeMarkers(markers);
        
        // Blend the result with the original color depth map
        cv::Mat wshed = markers_viz * 0.5 + color_depth * 0.5;

        cv::Mat bgr_dist_map_viz;
        cv::Mat color_depth_viz;

        cv::applyColorMap(bgr_dist_map, bgr_dist_map_viz, cv::COLORMAP_JET);
        cv::applyColorMap(color_depth, color_depth_viz, cv::COLORMAP_JET);

        cv::namedWindow("Depth Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Background", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Foreground extremums", cv::WINDOW_NORMAL);
        cv::namedWindow("Color Dist Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers distance", cv::WINDOW_NORMAL);
        cv::namedWindow("Sure Foreground", cv::WINDOW_NORMAL);
        cv::namedWindow("Color Depth Map", cv::WINDOW_NORMAL);
        cv::namedWindow("Markers", cv::WINDOW_NORMAL);
        cv::namedWindow("Watershed", cv::WINDOW_NORMAL);

        // You can also show intermediate steps
        cv::imshow("Depth Map", inverted_norm_depth);
        cv::imshow("Sure Background", sure_bg);
        cv::imshow("Sure Foreground extremums", sure_fg_extremums);
        cv::imshow("Color Dist Map", bgr_dist_map_viz);
        cv::imshow("Markers distance", dist_markers_viz);
        cv::imshow("Sure Foreground", sure_fg);
        cv::imshow("Color Depth Map", color_depth_viz);
        cv::imshow("Markers", markers_viz);
        cv::imshow("Watershed", wshed);

        cv::waitKey(0);

        std::vector<pcl::RGB> colorTable = generatePclColors(clusters.size());
        colorSegmentedPoints(_cloud, pcl::RGB(255,255,255));
        for(size_t i=0; i < clusters.size(); ++i) {
            colorSegmentedPoints(_cloud, clusters[i], colorTable[i]);
        }
    }
    
    return clusters;
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

double distanceToBoundingBoxSq(const pcl::PointXYZRGB& _point, const pcl_tools::BoundingBox& _bbox, bool is_2d = true)
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractClosestTree(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const std::vector<pcl::PointIndices>& _clusters,
    const pcl::PointXYZRGB& _point)
{
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> extracted_trees = extractClusters(_cloud, _clusters);

    if (extracted_trees.empty()) {
        return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr closest_tree = nullptr;
    double min_distance_sq = std::numeric_limits<double>::max();

    for (const auto& tree : extracted_trees)
    {
        pcl_tools::BoundingBox bbox = getBB(tree);
        double dist_sq = distanceToBoundingBoxSq(_point, bbox);

        if (dist_sq < min_distance_sq) {
            min_distance_sq = dist_sq;
            closest_tree = tree;
        }
    }

    return closest_tree;
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
    const float _searchRadius)
{
    pcl::PointIndices idx = findRadiusBoundary(
        _cloud,
        _boundaryIdx,
        _searchRadius
    );

    _cloud = extractPoints<pcl::PointXYZRGB>(_cloud, idx, false);
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

float computeDensity(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, float _radius)
{
    // Compute the volume of the sphere
    float volume = (4.0f / 3.0f) * M_PI * std::pow(_radius, 3);

    // Compute the density
    float density = _cloud->points.size() / volume;

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

    // Step 2: Create a KdTree and set the input cloud
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_cloud);

    std::vector<int> neighbor_indices;
    std::vector<float> neighbor_sq_dists;

    // --- ROBUSTNESS FIX ---
    // Add a check to ensure enough neighbors are found. Curvature estimation is unstable
    // with too few points and can cause the program to crash.
    if (kdtree->radiusSearch(_point, _radius, neighbor_indices, neighbor_sq_dists) < 8)
    {
        std::cout << "Error: Not enough points to compute curvature." << std::endl;
        // throw std::runtime_error("No curvature computed for the specified point.");
        return pcl::PrincipalCurvatures();
    }

    // Instead of copying the whole cloud, create a small cloud containing only the neighbors.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborhood(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*_cloud, neighbor_indices, *neighborhood);

    // Add the query point itself to the neighborhood cloud.
    neighborhood->push_back(_point);
    const int target_idx = neighborhood->size() - 1;

    // Step 3: Compute normals for the whole cloud
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormalsRad(neighborhood, _radius);

    // Step 4: Compute principal curvatures for the whole cloud
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> ce;
    ce.setInputCloud(neighborhood);
    ce.setInputNormals(normals);
    // ce.setSearchMethod(kdtree);
    ce.setRadiusSearch(_radius);

    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    ce.compute(*curvatures);

    // Step 5: Print out results
    const auto& curvature = curvatures->points[target_idx];
    float pc1 = curvature.pc1;
    float pc2 = curvature.pc2;
    float mean_curvature = (pc1 + pc2) / 2.0f;
    float gaussian_curvature = pc1 * pc2;

    std::cout << "Point " << target_idx << ": Principal Curvatures: " << pc1 << ", " << pc2
                << ", Mean Curvature: " << mean_curvature
                << ", Gaussian Curvature: " << gaussian_curvature << std::endl;
    std::cout << "  Principal Directions: (" << curvature.principal_curvature_x << ", "
                << curvature.principal_curvature_y << ", " << curvature.principal_curvature_z << ")" << std::endl;

    // Step 6: Return the curvature of the target point (last point in the cloud)
    if (!curvatures->empty()) {
        return curvatures->points[target_idx]; // The last point corresponds to the added point
    } else {
        std::cout << "Error: No curvature computed for the specified point." << std::endl;
        // throw std::runtime_error("No curvature computed for the specified point.");
        return pcl::PrincipalCurvatures();
    }
}

Eigen::Vector4f computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud)
{
    if (_cloud->empty()) {
        // throw std::runtime_error("Input cloud for computeCurvature cannot be empty.");
        std::cout << "Error: Input cloud for computePlane cannot be empty." << std::endl;
        return Eigen::Vector4f();
    }

    // Compute the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_cloud, centroid);

    // Perform PCA
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(_cloud);
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();
    // Eigen::Vector3f eigenvalues = pca.getEigenValues();

    // The eigenvector corresponding to the smallest eigenvalue is the normal of the plane
    Eigen::Vector3f normal = eigenvectors.col(2); // Third column (smallest eigenvalue)

    // Compute the plane coefficients (ax + by + cz + d = 0)
    float a = normal[0];
    float b = normal[1];
    float c = normal[2];
    float d = -(normal.dot(centroid.head<3>()));

    // Print the plane coefficients
    std::cout << "Plane coefficients (ax + by + cz + d = 0):\n";
    std::cout << "a: " << a << "\n";
    std::cout << "b: " << b << "\n";
    std::cout << "c: " << c << "\n";
    std::cout << "d: " << d << "\n";

    Eigen::Vector4f coefficients;
    coefficients[0] = a;
    coefficients[1] = b;
    coefficients[2] = c;
    coefficients[3] = d;

    return coefficients;
}

float computePlaneAngle(const Eigen::Vector4f& _coefficients)
{
    // Extract the normal vector (nx, ny, nz)
    Eigen::Vector3f normal(_coefficients[0], _coefficients[1], _coefficients[2]);

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

float pointToPlaneDistance(const pcl::PointXYZRGB& _point, const Eigen::Vector4f& _coefficients)
{
    float a = _coefficients[0];
    float b = _coefficients[1];
    float c = _coefficients[2];
    float d = _coefficients[3];

    // Compute the distance
    float distance = std::abs(a * _point.x + b * _point.y + c * _point.z + d);
    return distance;
}

float computeStandardDeviation(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const Eigen::Vector4f& coefficients)
{
    std::vector<float> distances;
    for (const auto& point : _cloud->points) {
        float distance = pointToPlaneDistance(point, coefficients);
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

std::vector<std::pair<float, float>> computeDistToPointsOfInterest(
    const pcl::PointXYZRGB& _landingPoint,
    const std::vector<pcl::PointXYZRGB>& _pointsOfInterest)
{
    std::vector<std::pair<float, float>> output;
    output.reserve(_pointsOfInterest.size());

    for (size_t i = 0; i < _pointsOfInterest.size(); ++i)
    {
        const pcl::PointXYZRGB pointOfInterest = _pointsOfInterest[i];

        float dist2D = computePointsDist2D(_landingPoint, pointOfInterest);
        float dist3D = computePointsDist3D(_landingPoint, pointOfInterest);

        output.emplace_back(dist2D, dist3D); 

        std::cout << "Distance to Point of interest #" << i << " 2D: " << output[i].first << std::endl;
        std::cout << "Distance to Point of interest #" << i << " 3D: " << output[i].second << std::endl;
    }

    return output;
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

bool checkInboundPoints(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _ogCloud, const std::vector<float>& _landing)
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

    if(_landing[0] < min_pt.x || _landing[0] > max_pt.x){
        std::cout << "\n\nWARNING: Target point out of bound in x: " << min_pt.x << " < " << _landing[0] << " < " << max_pt.x << "\n\n";
        isPointInBound = false;
    }
    if(_landing[1] < min_pt.y || _landing[1] > max_pt.y){
        std::cout << "\n\nWARNING: Target point out of bound in y: " << min_pt.y << " < " << _landing[1] << " < " << max_pt.y << "\n\n";
        isPointInBound = false;
    }

    if(_landing[2] < min_pt.z || _landing[2] > max_pt.z){
        std::cout << "\n\nWARNING: Target point out of bound in z: " << min_pt.z << " < " << _landing[2] << " < " << max_pt.z << "\n\n";
        isPointInBound = false;
    }

    return isPointInBound;
}

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centerClouds(
    const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& _clouds)
{
    // Create a vector to store the new, centered clouds
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centered_clouds;

    // Return an empty vector if the input is empty
    if (_clouds.empty() || !_clouds[0]) {
        return centered_clouds;
    }

    // 1. Compute the centroid of the first cloud to use as the origin
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_clouds[0], centroid);

    // 2. Create the transformation matrix to move the centroid to (0,0,0)
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];

    // 3. Loop through each cloud, apply the same transform, and store the result
    for (const auto& cloud : _clouds) {
        auto centered_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        if (cloud) { // Ensure the cloud pointer is valid
            pcl::transformPointCloud(*cloud, *centered_cloud, transform);
        }
        centered_clouds.push_back(centered_cloud);
    }

    return centered_clouds;
}

void addCloud2View(pcl::visualization::PCLVisualizer::Ptr _viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const std::string& _name)
{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    _viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, _name);
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _name);
}

void view(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> _clouds)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> centered_clouds = centerClouds(_clouds);
    int i = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud : centered_clouds) {
        addCloud2View(viewer, cloud, "cloud" + std::to_string(i));
        ++i;
    }
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    // https://github.com/PointCloudLibrary/pcl/issues/5237#issuecomment-1114255056
    // spin() instead of spinOnce() avoids crash
    viewer->spin();
}

}
