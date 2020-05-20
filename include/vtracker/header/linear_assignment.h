//#ifndef LINEAR_ASSIGNMENT_H
//#define LINEAR_ASSIGNMENT_H

//// STD
//#include <vector>

//// Eigen
//#include <eigen3/Eigen/Core>

//#include "header/kalman_filter.h"
//#include "header/track.h"
//#include "header/detection.h"
//#include "header/Hungarian.h"

//#define INFTY_COST 1e+5f


//class TrackDetectId 
//{
///** @brief
// *  Returns
// *  -------
// *  (List[(int, int)], List[int], List[int])
// *      Returns a tuple with the following three entries:
// *      * A list of matched track and detection indices.
// *      * A list of unmatched track indices.
// *      * A list of unmatched detection indices.
// */
//public:
//    TrackDetectId();
//    TrackDetectId( std::vector< cv::Point2i > _matches,
//                   std::vector< int > _unmatched_tracks,
//                   std::vector< int > _unmatched_detections );
    
//    std::vector< cv::Point2i > matches;
//    std::vector< int > unmatched_tracks;
//    std::vector< int > unmatched_detections;
    
//private:
    
//};
//TrackDetectId min_cost_matching( Eigen::ArrayXXf distance_metric( std::vector< Track > tracks, 
//                                                                  std::vector< Detection > dets, 
//                                                                  std::vector< int > track_indices, 
//                                                                  std::vector< int > detection_indices ),
//                                 float max_distance, 
//                                 std::vector< Track > tracks, 
//                                 std::vector< Detection > detections,
//                                 std::vector< int > track_indices = std::vector< int >(), 
//                                 std::vector< int > detection_indices = std::vector< int >() );
///** @brief
// *  Solve linear assignment problem.
// *  
// *  Parameters
// *  ----------
// *  distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
// *      The distance metric is given a list of tracks and detections as well as
// *      a list of N track indices and M detection indices. The metric should
// *      return the NxM dimensional cost matrix, where element (i, j) is the
// *      association cost between the i-th track in the given track indices and
// *      the j-th detection in the given detection_indices.
// *  max_distance : float
// *      Gating threshold. Associations with cost larger than this value are
// *      disregarded.
// *  tracks : List[track.Track]
// *      A list of predicted tracks at the current time step.
// *  detections : List[detection.Detection]
// *      A list of detections at the current time step.
// *  track_indices : List[int]
// *      List of track indices that maps rows in `cost_matrix` to tracks in
// *      `tracks` (see description above).
// *  detection_indices : List[int]
// *      List of detection indices that maps columns in `cost_matrix` to
// *      detections in `detections` (see description above).
// *  
// *  Returns
// *  -------
// *  (List[(int, int)], List[int], List[int])
// *      Returns a tuple with the following three entries:
// *      * A list of matched track and detection indices.
// *      * A list of unmatched track indices.
// *      * A list of unmatched detection indices.
// */

//TrackDetectId matching_cascade( Eigen::ArrayXXf gated_metric( std::vector< Track > tracks, 
//                                                              std::vector< Detection > dets, 
//                                                              std::vector< int > track_indices, 
//                                                              std::vector< int > detection_indices ),
//                                float max_distance, 
//                                int cascade_depth, 
//                                std::vector< Track > tracks, 
//                                std::vector< Detection > detections,
//                                std::vector< int > track_indices = std::vector< int >(), 
//                                std::vector< int > detection_indices = std::vector< int >() );
///** @brief
// *  Run matching cascade.
// *  
// *  Parameters
// *  ----------
// *  distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
// *      The distance metric is given a list of tracks and detections as well as
// *      a list of N track indices and M detection indices. The metric should
// *      return the NxM dimensional cost matrix, where element (i, j) is the
// *      association cost between the i-th track in the given track indices and
// *      the j-th detection in the given detection indices.
// *  max_distance : float
// *      Gating threshold. Associations with cost larger than this value are
// *      disregarded.
// *  cascade_depth: int
// *      The cascade depth, should be se to the maximum track age.
// *  tracks : List[track.Track]
// *      A list of predicted tracks at the current time step.
// *  detections : List[detection.Detection]
// *      A list of detections at the current time step.
// *  track_indices : Optional[List[int]]
// *      List of track indices that maps rows in `cost_matrix` to tracks in
// *      `tracks` (see description above). Defaults to all tracks.
// *  detection_indices : Optional[List[int]]
// *      List of detection indices that maps columns in `cost_matrix` to
// *      detections in `detections` (see description above). Defaults to all
// *      detections.
// *  
// *  Returns
// *  -------
// *  (List[(int, int)], List[int], List[int])
// *      Returns a tuple with the following three entries:
// *      * A list of matched track and detection indices.
// *      * A list of unmatched track indices.
// *      * A list of unmatched detection indices.
// */

//Eigen::ArrayXXf gate_cost_matrix( KalmanFilter kf, 
//                                  Eigen::ArrayXXf cost_matrix, 
//                                  std::vector< Track > tracks,
//                                  std::vector< Detection > detections,
//                                  std::vector< int > track_indices, 
//                                  std::vector< int > detection_indices, 
//                                  float gated_cost = INFTY_COST, 
//                                  bool only_position = false );
///** @brief
// *  Invalidate infeasible entries in cost matrix based on the state
// *  distributions obtained by Kalman filtering.
// *  
// *  Parameters
// *  ----------
// *  kf : The Kalman filter.
// *  cost_matrix : ndarray
// *      The NxM dimensional cost matrix, where N is the number of track indices
// *      and M is the number of detection indices, such that entry (i, j) is the
// *      association cost between `tracks[track_indices[i]]` and
// *      `detections[detection_indices[j]]`.
// *  tracks : List[track.Track]
// *      A list of predicted tracks at the current time step.
// *  detections : List[detection.Detection]
// *      A list of detections at the current time step.
// *  track_indices : List[int]
// *      List of track indices that maps rows in `cost_matrix` to tracks in
// *      `tracks` (see description above).
// *  detection_indices : List[int]
// *      List of detection indices that maps columns in `cost_matrix` to
// *      detections in `detections` (see description above).
// *  gated_cost : Optional[float]
// *      Entries in the cost matrix corresponding to infeasible associations are
// *      set this value. Defaults to a very large value.
// *  only_position : Optional[bool]
// *      If True, only the x, y position of the state distribution is considered
// *      during gating. Defaults to False.
// *  
// *  Returns
// *  -------
// *  ndarray
// *      Returns the modified cost matrix.
// */



//#endif // LINEAR_ASSIGNMENT_H
