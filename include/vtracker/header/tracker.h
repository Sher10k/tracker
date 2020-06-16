#ifndef TRACKER_H
#define TRACKER_H

// STD
#include <vector>

#include "kalman_filter.h"
#include "detection.h"
#include "nn_matching.h"
#include "track.h"
#include "Hungarian.h"


#define INFTY_COST 1e+10f

class TrackDetectId 
{
/** @brief
 *  Returns
 *  -------
 *  (List[(int, int)], List[int], List[int])
 *      Returns a tuple with the following three entries:
 *      * A list of matched track and detection indices.
 *      * A list of unmatched track indices.
 *      * A list of unmatched detection indices.
 */
public:
    TrackDetectId();
    TrackDetectId( std::vector< cv::Point2i > _matches,
                   std::vector< int > _unmatched_tracks,
                   std::vector< int > _unmatched_detections );
    
    std::vector< cv::Point2i > matches;
    std::vector< int > unmatched_tracks;
    std::vector< int > unmatched_detections;  
};

class Tracker
{
/** @brief
 *  This is the multi-target tracker.
 *  
 *  Parameters
 *  ----------
 *  metric : nn_matching.NearestNeighborDistanceMetric
 *     A distance metric for measurement-to-track association.
 *  max_age : int
 *      Maximum number of missed misses before a track is deleted.
 *  n_init : int
 *      Number of consecutive detections before the track is confirmed. The
 *      track state is set to `Deleted` if a miss occurs within the first
 *      `n_init` frames.
 *  
 *  Attributes
 *  ----------
 *  metric : nn_matching.NearestNeighborDistanceMetric
 *      The distance metric used for measurement to track association.
 *  max_age : int
 *      Maximum number of missed misses before a track is deleted.
 *  n_init : int
 *      Number of frames that a track remains in initialization phase.
 *  kf : kalman_filter.KalmanFilter
 *      A Kalman filter to filter target trajectories in image space.
 *  tracks : List[Track]
 *      The list of active tracks at the current time step.
 */
public:
    explicit Tracker( NearestNeighborDistanceMetric _metric,
                      float _max_iou_distance = 0.7f,
                      int _max_age = 30, 
                      int _n_init = 3 );
    
    NearestNeighborDistanceMetric metric;
    float max_iou_distance;
    int max_age;
    int n_init;
    KalmanFilter kf;
    std::vector< Track > tracks;
    
    void predict();
    /** @brief
     *  Propagate track state distributions one time step forward.
     *  
     * This function should be called once every time step, before `update`.
     */
    
    void update( std::vector< Detection > detections );
    /** @brief
     *  Perform measurement update and track management.
     *  
     *  Parameters
     *  ----------
     *  detections : List[deep_sort.detection.Detection]
     *      A list of detections at the current time step.
     */
    
    Eigen::ArrayXXf gate_cost_matrix( KalmanFilter kf, 
                                      Eigen::ArrayXXf cost_matrix, 
                                      std::vector< Track > tracks,
                                      std::vector< Detection > detections,
                                      std::vector< int > track_indices, 
                                      std::vector< int > detection_indices, 
                                      float gated_cost = INFTY_COST, 
                                      bool only_position = false );
    /** @brief
     *  Invalidate infeasible entries in cost matrix based on the state
     *  distributions obtained by Kalman filtering.
     *  
     *  Parameters
     *  ----------
     *  kf : The Kalman filter.
     *  cost_matrix : ndarray
     *      The NxM dimensional cost matrix, where N is the number of track indices
     *      and M is the number of detection indices, such that entry (i, j) is the
     *      association cost between `tracks[track_indices[i]]` and
     *      `detections[detection_indices[j]]`.
     *  tracks : List[track.Track]
     *      A list of predicted tracks at the current time step.
     *  detections : List[detection.Detection]
     *      A list of detections at the current time step.
     *  track_indices : List[int]
     *      List of track indices that maps rows in `cost_matrix` to tracks in
     *      `tracks` (see description above).
     *  detection_indices : List[int]
     *      List of detection indices that maps columns in `cost_matrix` to
     *      detections in `detections` (see description above).
     *  gated_cost : Optional[float]
     *      Entries in the cost matrix corresponding to infeasible associations are
     *      set this value. Defaults to a very large value.
     *  only_position : Optional[bool]
     *      If True, only the x, y position of the state distribution is considered
     *      during gating. Defaults to False.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      Returns the modified cost matrix.
     */
    
    Eigen::ArrayXXf gated_metric( std::vector< Track > tracks, 
                                  std::vector< Detection > detdetectionss, 
                                  std::vector< int > track_indices, 
                                  std::vector< int > detection_indices );
    
    Eigen::ArrayXf iou( cv::Rect2f bbox,
                        std::vector< cv::Rect2f > candidates );
    /** @brief
     *  Computer intersection over union.
     *  
     *  Parameters
     *  ----------
     *  bbox : ndarray
     *      A bounding box in format `(top left x, top left y, width, height)`.
     *  candidates : ndarray
     *      A matrix of candidate bounding boxes (one per row) in the same format
     *      as `bbox`.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      The intersection over union in [0, 1] between the `bbox` and each
     *      candidate. A higher score means a larger fraction of the `bbox` is
     *      occluded by the candidate.
     */
    
    Eigen::ArrayXXf iou_cost( std::vector< Track > tracks, 
                              std::vector< Detection > detections, 
                              std::vector< int > track_indices, 
                              std::vector< int > detection_indices );
    /** @brief
     *  An intersection over union distance metric.
     *  
     *  Parameters
     *  ----------
     *  tracks : List[deep_sort.track.Track]
     *      A list of tracks.
     *  detections : List[deep_sort.detection.Detection]
     *      A list of detections.
     *  track_indices : Optional[List[int]]
     *      A list of indices to tracks that should be matched. Defaults to
     *      all `tracks`.
     *  detection_indices : Optional[List[int]]
     *      A list of indices to detections that should be matched. Defaults
     *      to all `detections`.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      Returns a cost matrix of shape
     *      len(track_indices), len(detection_indices) where entry (i, j) is
     *      `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
     */
    
    TrackDetectId min_cost_matching_gm( float max_distance, 
                                        std::vector< Track > tracks, 
                                        std::vector< Detection > detections,
                                        std::vector< int > track_indices = std::vector< int >(), 
                                        std::vector< int > detection_indices = std::vector< int >() );
    TrackDetectId min_cost_matching_ic( float max_distance, 
                                        std::vector< Track > tracks, 
                                        std::vector< Detection > detections,
                                        std::vector< int > track_indices = std::vector< int >(), 
                                        std::vector< int > detection_indices = std::vector< int >() );
    /** @brief
     *  Solve linear assignment problem.
     *  
     *  Parameters
     *  ----------
     *  distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
     *      The distance metric is given a list of tracks and detections as well as
     *      a list of N track indices and M detection indices. The metric should
     *      return the NxM dimensional cost matrix, where element (i, j) is the
     *      association cost between the i-th track in the given track indices and
     *      the j-th detection in the given detection_indices.
     *  max_distance : float
     *      Gating threshold. Associations with cost larger than this value are
     *      disregarded.
     *  tracks : List[track.Track]
     *      A list of predicted tracks at the current time step.
     *  detections : List[detection.Detection]
     *      A list of detections at the current time step.
     *  track_indices : List[int]
     *      List of track indices that maps rows in `cost_matrix` to tracks in
     *      `tracks` (see description above).
     *  detection_indices : List[int]
     *      List of detection indices that maps columns in `cost_matrix` to
     *      detections in `detections` (see description above).
     *  
     *  Returns
     *  -------
     *  (List[(int, int)], List[int], List[int])
     *      Returns a tuple with the following three entries:
     *      * A list of matched track and detection indices.
     *      * A list of unmatched track indices.
     *      * A list of unmatched detection indices.
     */
    
    TrackDetectId matching_cascade( float max_distance, 
                                    int cascade_depth, 
                                    std::vector< Track > tracks, 
                                    std::vector< Detection > detections,
                                    std::vector< int > track_indices = std::vector< int >(), 
                                    std::vector< int > detection_indices = std::vector< int >() );
    /** @brief
     *  Run matching cascade.
     *  
     *  Parameters
     *  ----------
     *  distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
     *      The distance metric is given a list of tracks and detections as well as
     *      a list of N track indices and M detection indices. The metric should
     *      return the NxM dimensional cost matrix, where element (i, j) is the
     *      association cost between the i-th track in the given track indices and
     *      the j-th detection in the given detection indices.
     *  max_distance : float
     *      Gating threshold. Associations with cost larger than this value are
     *      disregarded.
     *  cascade_depth: int
     *      The cascade depth, should be se to the maximum track age.
     *  tracks : List[track.Track]
     *      A list of predicted tracks at the current time step.
     *  detections : List[detection.Detection]
     *      A list of detections at the current time step.
     *  track_indices : Optional[List[int]]
     *      List of track indices that maps rows in `cost_matrix` to tracks in
     *      `tracks` (see description above). Defaults to all tracks.
     *  detection_indices : Optional[List[int]]
     *      List of detection indices that maps columns in `cost_matrix` to
     *      detections in `detections` (see description above). Defaults to all
     *      detections.
     *  
     *  Returns
     *  -------
     *  (List[(int, int)], List[int], List[int])
     *      Returns a tuple with the following three entries:
     *      * A list of matched track and detection indices.
     *      * A list of unmatched track indices.
     *      * A list of unmatched detection indices.
     */
    
    void printTracker();
    void printTrack();
    
private:
    int _next_id;
    
    TrackDetectId _match( std::vector< Detection > & detections ); 
    void _initiate_track( Detection detection );
};

#endif // TRACKER_H
