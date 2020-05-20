#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

// STD
#include <iostream>
#include <vector>

// CV
#include <opencv2/core.hpp>

// EIGEN
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

/** @brief
 *  Table for the 0.95 quantile of the chi-square distribution with N degrees of
 *  freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
 *  function and used as Mahalanobis gating threshold.
 */
static float chi2inv95[9] = {
    3.8415f,
    5.9915f,
    7.8147f,
    9.4877f,
    11.070f,
    12.592f,
    14.067f,
    15.507f,
    16.919f
};


class KalmanFilter
{
/** @brief
 *     A simple Kalman filter for tracking bounding boxes in image space.
 * 
 *  The 8-dimensional state space
 * 
 *      x, y, a, h, vx, vy, va, vh
 * 
 *  contains the bounding box center position (x, y), aspect ratio a, height h,
 *  and their respective velocities.
 * 
 *  Object motion follows a constant velocity model. The bounding box location
 *  (x, y, a, h) is taken as direct observation of the state space (linear
 *  observation model).
 */
public:
    explicit KalmanFilter();
    
    void initiate( cv::Rect2f measurement );
    /** @brief 
     *  Create track from unassociated measurement.
     *  
     *  Parameters
     *  ----------
     *  measurement : ndarray
     *      Bounding box coordinates (x, y, a, h) with center position (x, y),
     *      aspect ratio a, and height h.
     *  
     *  Returns
     *  -------
     *  (ndarray, ndarray)
     *      Returns the mean vector (8 dimensional) and covariance matrix (8x8
     *      dimensional) of the new track. Unobserved velocities are initialized
     *      to 0 mean.
     */
    
    void predict( Eigen::Matrix< float, 1, 8 > & _mean,
                  Eigen::Matrix< float, 8, 8 > & _covariance );
    /** @brief 
     *  Run Kalman filter prediction step.
     *  
     *  Parameters
     *  ----------
     *  mean : ndarray
     *      The 8 dimensional mean vector of the object state at the previous
     *      time step.
     *  covariance : ndarray
     *      The 8x8 dimensional covariance matrix of the object state at the
     *      previous time step.
     *  
     *  Returns
     *  -------
     *  (ndarray, ndarray)
     *      Returns the mean vector and covariance matrix of the predicted
     *      state. Unobserved velocities are initialized to 0 mean.
     */
    
    void project( Eigen::Matrix< float, 1, 8 > & _mean,
                  Eigen::Matrix< float, 8, 8 > & _covariance );
    /** @brief 
     *  Project state distribution to measurement space.
     *  
     *  Parameters
     *  ----------
     *  mean : ndarray
     *      The state's mean vector (8 dimensional array).
     *  covariance : ndarray
     *      The state's covariance matrix (8x8 dimensional).
     *  
     *  Returns
     *  -------
     *  (ndarray, ndarray)
     *      Returns the projected mean and covariance matrix of the given state
     *      estimate.
     */
    
    void update( Eigen::Matrix< float, 1, 8 > & _mean,
                 Eigen::Matrix< float, 8, 8 > & _covariance,
                 cv::Rect2f _measurement );
    /** @brief 
     *  Run Kalman filter correction step.
     *  
     *  Parameters
     *  ----------
     *  mean : ndarray
     *      The predicted state's mean vector (8 dimensional).
     *  covariance : ndarray
     *      The state's covariance matrix (8x8 dimensional).
     *  measurement : ndarray
     *      The 4 dimensional measurement vector (x, y, a, h), where (x, y)
     *      is the center position, a the aspect ratio, and h the height of the
     *      bounding box.
     *  
     *  Returns
     *  -------
     *  (ndarray, ndarray)
     *      Returns the measurement-corrected state distribution.
     */
    
    std::vector< float > gating_distance( Eigen::Matrix< float, 1, 8 > & _mean,
                                          Eigen::Matrix< float, 8, 8 > & _covariance,
                                          std::vector< cv::Rect2f > _measurements,
                                          bool only_position );
    /** @brief 
     *  Compute gating distance between state distribution and measurements.
     *  
     *  A suitable distance threshold can be obtained from `chi2inv95`. If
     *  `only_position` is False, the chi-square distribution has 4 degrees of
     *  freedom, otherwise 2.
     *  
     *  Parameters
     *  ----------
     *  mean : ndarray
     *      Mean vector over the state distribution (8 dimensional).
     *  covariance : ndarray
     *      Covariance of the state distribution (8x8 dimensional).
     *  measurements : ndarray
     *      An Nx4 dimensional matrix of N measurements, each in
     *      format (x, y, a, h) where (x, y) is the bounding box center
     *      position, a the aspect ratio, and h the height.
     *  only_position : Optional[bool]
     *      If True, distance computation is done with respect to the bounding
     *      box center position only.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      Returns an array of length N, where the i-th element contains the
     *      squared Mahalanobis distance between (mean, covariance) and
     *      `measurements[i]`.
     */
    
    Eigen::MatrixXf get_mean();
    Eigen::MatrixXf get_covariance();
    Eigen::MatrixXf get_proj_mean();
    Eigen::MatrixXf get_proj_covariance();
    
private:
    Eigen::Matrix< float, 8, 8 > _motion_mat;    // [ 8 x 8 ]
    Eigen::Matrix< float, 4, 8 > _update_mat;    // [ 4 x 8 ]
    
    float _std_weight_position;
    float _std_weight_velocity;
    
    Eigen::Matrix< float, 1, 8 > mean;              // [ 1 x 8 ]
    Eigen::Matrix< float, 8, 8 > covariance;        // [ 8 x 8 ]
    Eigen::Matrix< float, 1, 4 > projected_mean;    // [ 1 x 8 ]
    Eigen::Matrix< float, 4, 4 > projected_cov;     // [ 4 x 4 ]
};

#endif // KALMAN_FILTER_H
