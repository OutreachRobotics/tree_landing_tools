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

template <typename PointT>
void extractPoints(
    const pcl::PointCloud<PointT>& _ogCloud,
    pcl::PointCloud<PointT>& _outputCloud,
    const pcl::PointIndices& _indices,
    bool _isExtractingOutliers)
{
    pcl::copyPointCloud(_ogCloud, _outputCloud);

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(_ogCloud.makeShared());
    extract.setNegative(_isExtractingOutliers);
    extract.setIndices(std::make_shared<pcl::PointIndices>(_indices));
    extract.filter(_outputCloud);
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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
    extractPoints(*_pointCloud, *_pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointIndices computeNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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
    extractPoints(*_pointCloud, *_pointCloud, removedIdx, true);

    return removedIdx;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const int _nNeighborsSearch)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsPC(_pointCloud, normalsCloud, _nNeighborsSearch);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const int _nNeighborsSearch,
    const pcl::PointXYZRGB& _centroid)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsPC(_pointCloud, normalsCloud, _nNeighborsSearch, _centroid);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const float _radiusSearch)
{
    pcl::PointCloud<pcl::PointNormal>::Ptr normalsCloud(new pcl::PointCloud<pcl::PointNormal>);
    computeNormalsRadPC(_pointCloud, normalsCloud, _radiusSearch);
    return normalsCloud;
}

pcl::PointCloud<pcl::PointNormal>::Ptr extractNormalsRadPC(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractNeighborPC(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const pcl::PointXYZRGB& _center,
    const float _radius)
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(_pointCloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborsPC(new pcl::PointCloud<pcl::PointXYZRGB>);
    neighborsPC->points.reserve(_pointCloud->points.size());

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    if(kdtree->radiusSearch(_center, _radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0){
        for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i){
            neighborsPC->points.emplace_back(_pointCloud->points[pointIdxRadiusSearch[i]]);
        }
    }

    return neighborsPC;
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

std::vector<pcl::PointIndices> extractClusters(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
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
        extractPoints(*_pointCloud, *_pointCloud, concatenateClusters(cluster_indices), false);
    }

    return cluster_indices;
}

pcl::PointIndices extractBiggestSegment(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    std::vector<pcl::PointIndices> _segment_indices)
{
    int largest_cluster_index = -1;
    int largest_cluster_size = 0;
    for (size_t i = 0; i < _segment_indices.size(); ++i)
    {
        int cluster_size = _segment_indices[i].indices.size();
        if (cluster_size > largest_cluster_size)
        {
            largest_cluster_index = i;
            largest_cluster_size = cluster_size;
        }
    }

    pcl::PointIndices inliers;
    if (largest_cluster_index != -1)
    {
        inliers = _segment_indices[largest_cluster_index];
        extractPoints(*_pointCloud, *_pointCloud, inliers, false);
        std::cout << "The point cloud has " << std::to_string(_segment_indices.size()) << " clusters." << std::endl;
    }
    else
    {
        std::cout << "No clusters found in the point cloud." << std::endl;
    } 

    return inliers;
}

pcl::PointIndices extractBiggestCluster(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _pointCloud,
    const float _threshold,
    const int _minPoints)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tempPointCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_pointCloud));

    // Cluster extraction
    std::vector<pcl::PointIndices> cluster_indices = extractClusters(
        tempPointCloud,
        _threshold,
        _minPoints
    );

    return extractBiggestSegment(_pointCloud, cluster_indices);
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
    const int _kernelSize)
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
    cv::medianBlur(depth_map, filtered_depth_map, _kernelSize);
    return {filtered_depth_map, grid, min_x, min_y};
}

std::vector<pcl::PointIndices> mapSegmentsToIndices(
    const cv::Mat& _markers,
    const DepthMapData& _mapData)
{
    // Use a map internally to associate labels with their index clusters
    std::map<int, pcl::PointIndices::Ptr> temp_segment_map;

    for (int v = 0; v < _markers.rows; ++v) {
        for (int u = 0; u < _markers.cols; ++u) {
            int label = _markers.at<int>(v, u);

            // This already skips the background cluster (label <= 1)
            if (label <= 1) {
                continue;
            }

            // If this is the first point for a new segment, create an entry in the map
            if (temp_segment_map.find(label) == temp_segment_map.end()) {
                temp_segment_map[label] = pcl::make_shared<pcl::PointIndices>();
            }

            // Find the original point index from the grid map
            int grid_x = u + _mapData.min_x;
            int grid_y = v + _mapData.min_y;
            auto it = _mapData.grid.find({grid_x, grid_y});

            if (it != _mapData.grid.end()) {
                // Add the index to the appropriate cluster
                temp_segment_map[label]->indices.push_back(it->second);
            }
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmentWatershed(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _leafSize,
    const int _kernelSize,
    const float _thresh_fg)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));

    DepthMapData depthMapData = computeDepthMap( _cloud, _leafSize, _kernelSize);

    // 1. NORMALIZE AND COLORIZE THE DEPTH MAP
    // Convert the 32F depth map to a visual 8U format
    cv::Mat norm_depth;
    cv::normalize(depthMapData.depthMap, norm_depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Create a 3-channel color image for the watershed algorithm
    cv::Mat color_depth;
    cv::applyColorMap(norm_depth, color_depth, cv::COLORMAP_JET);

    // 2. CREATE MARKERS AUTOMATICALLY
    // Threshold to get a binary image of potential objects
    // cv::Mat binary;
    // cv::threshold(norm_depth, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Define "sure background" as all pixels with 0 value in the original depth map
    cv::Mat sure_bg;
    cv::compare(depthMapData.depthMap, 0, sure_bg, cv::CMP_EQ);

    // Create an inverted version of the background mask
    cv::Mat foreground_mask;
    cv::bitwise_not(sure_bg, foreground_mask);

    // Find "sure foreground" area using distance transform
    cv::Mat dist_transform;
    // cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 5);
    cv::distanceTransform(foreground_mask, dist_transform, cv::DIST_L2, 5);
    cv::normalize(dist_transform, dist_transform, 0, 1.0, cv::NORM_MINMAX);

    cv::Mat sure_fg;
    cv::threshold(dist_transform, sure_fg, _thresh_fg, 1.0, cv::THRESH_BINARY);
    sure_fg.convertTo(sure_fg, CV_8U, 255.0);

    // Find unknown region
    // Combine the known regions into a single mask
    cv::Mat known_markers;
    cv::bitwise_or(sure_bg, sure_fg, known_markers);

    // The unknown region is everything not in the known markers
    cv::Mat unknown;
    cv::bitwise_not(known_markers, unknown);

    // 3. LABEL THE MARKERS
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);
    // Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1;
    // Mark the region of unknown with zero
    markers.setTo(0, unknown == 255);

    // 4. APPLY WATERSHED
    cv::watershed(color_depth, markers);

    // 5. VISUALIZE THE RESULT
    cv::Mat markers_viz = visualizeMarkers(markers);
    
    // Blend the result with the original color depth map
    cv::Mat wshed = markers_viz * 0.5 + color_depth * 0.5;

    // You can also show intermediate steps
    cv::imshow("Depth Map", norm_depth);
    cv::imshow("Color Depth Map", color_depth);
    cv::imshow("Sure Foreground", sure_fg);
    cv::imshow("Sure Background", sure_bg);
    cv::imshow("Unknown", unknown);
    cv::imshow("Markers", markers_viz);
    cv::imshow("Watershed", wshed);
    cv::waitKey(0);

    std::vector<pcl::PointIndices> clusters = mapSegmentsToIndices(markers, depthMapData);
    extractBiggestSegment(_cloud, clusters);

    std::vector<pcl::RGB> colorTable = generatePclColors(clusters.size());

    colorSegmentedPoints(outputCloud, pcl::RGB(255,255,255));
    for(size_t i=0; i < clusters.size(); ++i) {
        colorSegmentedPoints(outputCloud, clusters[i], colorTable[i]);
    }
    
    return outputCloud;
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
    mls.setPolynomialOrder(2);
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
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointCloud<pcl::PointNormal>::Ptr _normalsCloud,
    const int _searchNeighbors)
{
    pcl::PointIndices idx = findBoundary(
        _cloud,
        _normalsCloud,
        _searchNeighbors
    );

    extractPoints(*_cloud, *_cloud, idx, false);
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
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointIndices _boundaryIdx,
    const float _searchRadius)
{
    pcl::PointIndices idx = findRadiusBoundary(
        _cloud,
        _boundaryIdx,
        _searchRadius
    );

    extractPoints(*_cloud, *_cloud, idx, false);
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
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _leafSize)
{
    pcl::PointIndices idx = findSurface(
        _cloud,
        _leafSize
    );

    extractPoints(*_cloud, *_cloud, idx, false);
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

int projectPoint(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    pcl::PointXYZRGB& _point)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    for (auto& point : cloud_copy->points) {
        point.z = 0;
    }

    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cloud_copy);

    std::vector<int> point_indices;
    std::vector<float> point_distances;
    int nClosestPoints = 4;
    float avgZ = 0.0;
    if (kdtree->nearestKSearch(_point, nClosestPoints, point_indices, point_distances) > 0) {
        for (auto& idx : point_indices) {
            pcl::PointXYZRGB closest_point = _cloud->points[idx];
            std::cout << "Closest point: (" << closest_point.x << ", "
                      << closest_point.y << ", " << closest_point.z << ")" << std::endl;
            std::cout << "Z coordinate: " << closest_point.z << std::endl;
            avgZ += closest_point.z;
        }
        avgZ = avgZ/float(point_indices.size());
        std::cout << "Z average coordinate: " << avgZ << std::endl;
    }

    _point.z = avgZ;

    return avgZ;
}

pcl::PrincipalCurvatures computeCurvature(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const pcl::PointXYZRGB& _point,
    const float _radius)
{
    // Step 1: Create a copy of the input cloud and add the point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*_cloud));
    cloud_copy->push_back(_point);

    // Step 2: Create a KdTree and set the input cloud
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZRGB>);
    kdtree->setInputCloud(cloud_copy);

    // Step 3: Compute normals for the whole cloud
    pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormalsRad(cloud_copy, _radius);

    // Step 4: Compute principal curvatures for the whole cloud
    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::PrincipalCurvatures> ce;
    ce.setInputCloud(cloud_copy);
    ce.setInputNormals(normals);
    ce.setSearchMethod(kdtree);
    ce.setRadiusSearch(_radius);

    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    ce.compute(*curvatures);

    // Step 5: Print out results
    int lastIdx = curvatures->size() - 1;
    const auto& curvature = curvatures->points[lastIdx];
    float pc1 = curvature.pc1;
    float pc2 = curvature.pc2;
    float mean_curvature = (pc1 + pc2) / 2.0f;
    float gaussian_curvature = pc1 * pc2;

    std::cout << "Point " << lastIdx << ": Principal Curvatures: " << pc1 << ", " << pc2
                << ", Mean Curvature: " << mean_curvature
                << ", Gaussian Curvature: " << gaussian_curvature << std::endl;
    std::cout << "  Principal Directions: (" << curvature.principal_curvature_x << ", "
                << curvature.principal_curvature_y << ", " << curvature.principal_curvature_z << ")\n";

    // Step 6: Return the curvature of the target point (last point in the cloud)
    if (!curvatures->empty()) {
        return curvatures->points[lastIdx]; // The last point corresponds to the added point
    } else {
        throw std::runtime_error("No curvature computed for the specified point.");
    }
}

Eigen::Vector4f computePlane(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud)
{
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
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud,
    const float _radius,
    const bool _isMin)
{
    pcl::PointIndices idx = findLocalExtremums(_cloud, _radius, _isMin);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr localExtremums(new pcl::PointCloud<pcl::PointXYZRGB>);
    extractPoints(*_cloud, *localExtremums, idx, false);
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

bool checkInboundPoints(const pcl::PointXYZRGB _min_pt, const pcl::PointXYZRGB _max_pt, float& _x, float& _y)
{
    bool isPointInBound = true;
    // Bounding box dimensions
    float width = _max_pt.x - _min_pt.x;
    float height = _max_pt.y - _min_pt.y;
    float depth = _max_pt.z - _min_pt.z;
    float c_x = width / 2.0 + _min_pt.x;
    float c_y = height / 2.0 + _min_pt.y;

    // std::cout << "\n" << "AABB Dimensions: "
    //         << width << " (W) x "
    //         << height << " (H) x "
    //         << depth << " (D)\n\n";

    if(_x < _min_pt.x || _x > _max_pt.x){
        std::cout << "\n\nWARNING: Target point out of bound: " << _min_pt.x << " < " << _x << " < " << _max_pt.x << "\n\n";
        isPointInBound = false;
    }
    if(_y < _min_pt.y || _y > _max_pt.y){
        std::cout << "\n\nWARNING: Target point out of bound: " << _min_pt.y << " < " << _y << " < " << _max_pt.y << "\n\n";
        isPointInBound = false;
    }

    return isPointInBound;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr centerCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud)
{
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*_cloud, centroid);

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << -centroid[0], -centroid[1], -centroid[2];

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_centered(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*_cloud, *cloud_centered, transform);
    return cloud_centered;
}

void addCloud2View(pcl::visualization::PCLVisualizer::Ptr _viewer, pcl::PointCloud<pcl::PointXYZRGB>::Ptr _cloud, const std::string& _name)
{
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(_cloud);
    _viewer->addPointCloud<pcl::PointXYZRGB> (_cloud, rgb, _name);
    _viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, _name);
}

void view(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> _clouds)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    int i = 0;
    for (pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud : _clouds) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr centeredCloud = centerCloud(cloud);
        addCloud2View(viewer, centeredCloud, "cloud" + std::to_string(i));
        ++i;
    }
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    // https://github.com/PointCloudLibrary/pcl/issues/5237#issuecomment-1114255056
    // spin() instead of spinOnce() avoids crash
    viewer->spin();
}

}
