#ifndef TRACK_H
#define TRACK_H

// STD
#include <iostream>
#include <vector>

// EIGEN
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// CV
#include <opencv2/core.hpp>

#include "kalman_filter.h"
#include "detection.h"


enum TrackState
{
/** @brief
 *  Enumeration type for the single target track state. Newly created tracks are
 *  classified as `tentative` until enough evidence has been collected. Then,
 *  the track state is changed to `confirmed`. Tracks that are no longer alive
 *  are classified as `deleted` to mark them for removal from the set of active
 *  tracks.
 */
    Tentative = 1,
    Confirmed = 2,
    Deleted = 3
};


class Track
{
/** @brief
 *  A single target track with state space `(x, y, a, h)` and associated
 *  velocities, where `(x, y)` is the center of the bounding box, `a` is the
 *  aspect ratio and `h` is the height.
 *  
 *  Parameters
 *  ----------
 *  mean : ndarray
 *      Mean vector of the initial state distribution.
 *  covariance : ndarray
 *      Covariance matrix of the initial state distribution.
 *  track_id : int
 *      A unique track identifier.
 *  n_init : int
 *      Number of consecutive detections before the track is confirmed. The
 *      track state is set to `Deleted` if a miss occurs within the first
 *      `n_init` frames.
 *  max_age : int
 *      The maximum number of consecutive misses before the track state is
 *      set to `Deleted`.
 *  feature : Optional[ndarray]
 *      Feature vector of the detection this track originates from. If not None,
 *      this feature is added to the `features` cache.
 *  
 *  Attributes
 *  ----------
 *  mean : ndarray
 *      Mean vector of the initial state distribution.
 *  covariance : ndarray
 *      Covariance matrix of the initial state distribution.
 *  track_id : int
 *      A unique track identifier.
 *  hits : int
 *      Total number of measurement updates.
 *  age : int
 *      Total number of frames since first occurance.
 *  time_since_update : int
 *      Total number of frames since last measurement update.
 *  state : TrackState
 *      The current track state.
 *  features : List[ndarray]
 *      A cache of features. On each measurement update, the associated feature
 *      vector is added to this list.
 */
public:
    explicit Track( Eigen::Matrix< float, 1, 8 > _mean, 
                    Eigen::Matrix< float, 8, 8 > _covariance, 
                    int _track_id, 
                    int __n_init, 
                    int __max_age,
                    std::vector< float > _feature = std::vector< float >() );
    
    Eigen::Matrix< float, 1, 8 > mean;              // [ 1 x 8 ]
    Eigen::Matrix< float, 8, 8 > covariance;        // [ 8 x 8 ]
    int track_id;
    int hits;
    int age;
    int time_since_update;
    TrackState state;
    std::vector< std::vector< float > > features;
    
    cv::Rect2f to_tlwh();
    /** @brief
     *  Get current position in bounding box format `(top left x, top left y,
     *  width, height)`.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      The bounding box.
     */
    
    cv::Rect2f to_tlbr();
    /** @brief
     *  Get current position in bounding box format `(min x, miny, max x,
     *  max y)`.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      The bounding box.
     */
    
    void predict( KalmanFilter & kf );
    /** @brief
     *  Propagate the state distribution to the current time step using a
     *  Kalman filter prediction step.
     *  
     *  Parameters
     *  ----------
     *  kf : kalman_filter.KalmanFilter
     *      The Kalman filter.
     */
    
    void update( KalmanFilter kf, Detection detection );
    /** @brief
     *  Perform Kalman filter measurement update step and update the feature
     *  cache.
     *  
     *  Parameters
     *  ----------
     *  kf : kalman_filter.KalmanFilter
     *      The Kalman filter.
     *  detection : Detection
     *      The associated detection.
     */
    
    void mark_missed();
    /** @brief
     *  Mark this track as missed (no association at the current time step).
     */
    
    bool is_tentative();
    /** @brief
     *  Returns True if this track is tentative (unconfirmed).
     */
    
    bool is_confirmed();
    /** @brief
     *  Returns True if this track is confirmed.
     */
    
    bool is_deleted();
    /** @brief
     *  Returns True if this track is dead and should be deleted. 
     */
    
private:
    int _n_init;
    int _max_age;
    
};

    
#endif // TRACK_H
