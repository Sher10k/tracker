#include "detection.h"

Detection::Detection( cv::Rect2f _bbox,
                      float _confidence,
                      std::vector< float > _feature )
    : tlwh(_bbox), confidence(_confidence), feature(_feature) {}

cv::Rect2f Detection::to_tlbr()
{
    cv::Rect2f ret = this->tlwh;
    ret.width += ret.x;
    ret.height += ret.y;
    return ret;
}

cv::Rect2f Detection::to_xyah()
{
    cv::Rect2f ret = this->tlwh;
    ret.x += ret.width / 2.0f;
    ret.y += ret.height / 2.0f;
    ret.width /= ret.height;
    return ret;
}
