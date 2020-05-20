#ifndef DETECTION_H
#define DETECTION_H

// STD
#include <vector>

// CV
#include <opencv2/core.hpp>


class Detection
{
/** @brief
 *     This class represents a bounding box detection in a single image.
 *  
 *  Parameters
 *  ----------
 *  tlwh : array_like
 *      Bounding box in format `(x, y, w, h)`.
 *  confidence : float
 *      Detector confidence score.
 *  feature : array_like
 *      A feature vector that describes the object contained in this image.
 *  
 *  Attributes
 *  ----------
 *  tlwh : ndarray
 *      Bounding box in format `(top left x, top left y, width, height)`.
 *  confidence : ndarray
 *      Detector confidence score.
 *  feature : ndarray | NoneType
 *      A feature vector that describes the object contained in this image.
 */
public:
    explicit Detection( cv::Rect2f _bbox = cv::Rect2f(0, 0, 0, 0),
                        float _confidence = 0,
                        std::vector< float > _feature = std::vector< float >() );
    
    cv::Rect2f tlwh;
    float confidence;
    std::vector< float > feature;
    
    cv::Rect2f to_tlbr();
    /** @brief
     *  Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
     *  `(top left, bottom right)`.
     */
    
    cv::Rect2f to_xyah();
    /** @brief
     *  Convert bounding box to format `(center x, center y, aspect ratio,
     *  height)`, where the aspect ratio is `width / height`.
     */
};


#endif // DETECTION_H
