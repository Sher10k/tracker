#ifndef NN_MATCHING_H
#define NN_MATCHING_H

// STD
#include <iostream>
#include <vector>
#include <queue>
#include <map>

// CV
#include <opencv2/core.hpp>

// Eigen
#include <eigen3/Eigen/Core>
//#include <eigen3/Eigen/Dense>


Eigen::ArrayXXf _pdist( std::vector< std::vector< float > > a, 
                        std::vector< std::vector< float > > b );        // return value ????????????????
/** @brief
 *  Compute pair-wise squared distance between points in `a` and `b`.
 *   
 *  Parameters
 *  ----------
 *  a : array_like
 *      An NxM matrix of N samples of dimensionality M.
 *  b : array_like
 *      An LxM matrix of L samples of dimensionality M.
 *  
 *  Returns
 *  -------
 *  ndarray
 *      Returns a matrix of size len(a), len(b) such that eleement (i, j)
 *      contains the squared distance between `a[i]` and `b[j]`.
 */

Eigen::ArrayXXf _cosine_distance( std::vector< std::vector< float > > a, 
                                  std::vector< std::vector< float > > b,
                                  bool data_is_normalized = false );    // return value ????????????????
/** @brief
 *  Compute pair-wise cosine distance between points in `a` and `b`.
 *  
 *  Parameters
 *  ----------
 *  a : array_like
 *      An NxM matrix of N samples of dimensionality M.
 *  b : array_like
 *      An LxM matrix of L samples of dimensionality M.
 *  data_is_normalized : Optional[bool]
 *      If True, assumes rows in a and b are unit length vectors.
 *      Otherwise, a and b are explicitly normalized to lenght 1.
 *  
 *  Returns
 *  -------
 *  ndarray
 *      Returns a matrix of size len(a), len(b) such that eleement (i, j)
 *      contains the squared distance between `a[i]` and `b[j]`.
 */

Eigen::ArrayXf _nn_euclidean_distance( std::vector< std::vector< float > > x, 
                                       std::vector< std::vector< float > > y );
/** @brief
 *  Helper function for nearest neighbor distance metric (Euclidean).
 *  
 *  Parameters
 *  ----------
 *  x : ndarray
 *      A matrix of N row-vectors (sample points).
 *  y : ndarray
 *      A matrix of M row-vectors (query points).
 *
 *  Returns
 *  -------
 *  ndarray
 *      A vector of length M that contains for each entry in `y` the
 *      smallest Euclidean distance to a sample in `x`.
 */
 
Eigen::ArrayXf _nn_cosine_distance( std::vector< std::vector< float > > x, 
                                    std::vector< std::vector< float > > y );
/** @brief
 *  Helper function for nearest neighbor distance metric (cosine).
 *  
 *  Parameters
 *  ----------
 *  x : ndarray
 *      A matrix of N row-vectors (sample points).
 *  y : ndarray
 *      A matrix of M row-vectors (query points).
 *  
 *  Returns
 *  -------
 *  ndarray
 *      A vector of length M that contains for each entry in `y` the
 *      smallest cosine distance to a sample in `x`.
 */


class NearestNeighborDistanceMetric
{
/** @brief
 *  A nearest neighbor distance metric that, for each target, returns
 *  the closest distance to any sample that has been observed so far.
 *  
 *  Parameters
 *  ----------
 *  metric : str
 *      Either "euclidean" or "cosine".
 *  matching_threshold: float
 *      The matching threshold. Samples with larger distance are considered an
 *      invalid match.
 *  budget : Optional[int]
 *      If not None, fix samples per class to at most this number. Removes
 *      the oldest samples when the budget is reached.
 *  
 *  Attributes
 *  ----------
 *  samples : Dict[int -> List[ndarray]]
 *      A dictionary that maps from target identities to the list of samples
 *      that have been observed so far.
 */
public:
    explicit NearestNeighborDistanceMetric( std::string _metric = "cosine",
                                            float _matching_threshold = 0.2f,
                                            unsigned _budget = 0 );
    
    Eigen::ArrayXf operator()( std::vector< std::vector< float > > x, 
                                std::vector< std::vector< float > > y );        // НУЖНА ЛИ ОНА ???????
    
    std::string metric;
    float matching_threshold;
    unsigned budget;
    std::map< int, std::deque< std::vector< float > > > samples;
    
    void partial_fit( std::vector< std::vector< float > > features, 
                      std::vector< int > targets, 
                      std::vector< int > active_targets );
    /** @brief
     *  Update the distance metric with new data.
     *       
     *  Parameters
     *  ----------
     *  features : ndarray
     *      An NxM matrix of N features of dimensionality M.
     *  targets : ndarray
     *      An integer array of associated target identities.
     *  active_targets : List[int]
     *      A list of targets that are currently present in the scene.
     */
    
    Eigen::ArrayXXf distance( std::vector< std::vector< float > > features, 
                              std::vector< int > targets );
    /** @brief
     *  Compute distance between features and targets.
     *  
     *  Parameters
     *  ----------
     *  features : ndarray
     *      An NxM matrix of N features of dimensionality M.
     *  targets : List[int]
     *      A list of targets to match the given `features` against.
     *  
     *  Returns
     *  -------
     *  ndarray
     *      Returns a cost matrix of shape len(targets), len(features), where
     *      element (i, j) contains the closest squared distance between
     *      `targets[i]` and `features[j]`.
     */
    
private:
    
};

#endif // NN_MATCHING_H
