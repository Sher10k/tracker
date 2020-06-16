#include "../header/track.h"

Track::Track( Eigen::Matrix< float, 1, 8 > _mean, 
              Eigen::Matrix< float, 8, 8 > _covariance, 
              int _track_id, 
              int __n_init, 
              int __max_age,
              std::vector< float > _feature )
    : mean(_mean),
      covariance(_covariance),
      track_id(_track_id),
      hits(1),
      age(1),
      time_since_update(0),
      state(TrackState::Tentative),
      _n_init(__n_init),
      _max_age(__max_age)
{
    if ( !_feature.empty() )
        features.push_back(_feature);
}

cv::Rect2f Track::to_tlwh()
{
    cv::Rect2f ret = cv::Rect2f( this->mean(0, 0), 
                                 this->mean(0, 1), 
                                 this->mean(0, 2), 
                                 this->mean(0, 3) );
    ret.width *= ret.height;
    ret.x -= ret.width / 2;
    ret.y -= ret.height / 2;
    return ret;
}

cv::Rect2f Track::to_tlbr()
{
    cv::Rect2f ret = this->to_tlwh();
    ret.width += ret.x;
    ret.height += ret.y;
    return ret;
}

void Track::predict( KalmanFilter & kf )
{
//    std::cout << "// --- mean_1: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_1: \n" << this->covariance << "\n";
    kf.predict( this->mean, this->covariance );
    this->mean = kf.get_mean();
    this->covariance = kf.get_covariance();
//    std::cout << "// --- mean_2: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_2: \n" << this->covariance << "\n\n";
    
    this->age++;
    this->time_since_update++;
}

void Track::update( KalmanFilter kf, Detection detection )
{
//    std::cout << "// --- mean_1: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_1: \n" << this->covariance << "\n";
    kf.update( this->mean, this->covariance, detection.to_xyah() );     // --- ERROR 0.0.4
    this->mean = kf.get_mean();
    this->covariance = kf.get_covariance();
    this->features.push_back( detection.feature );
//    std::cout << "// --- mean_2: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_2: \n" << this->covariance << "\n\n";
    
    this->hits++;
    this->time_since_update = 0;
    if ( (this->state == TrackState::Tentative) && (this->hits >= this->_n_init) )
        this->state = TrackState::Confirmed;
}

void Track::mark_missed()
{
    if ( this->state == TrackState::Tentative )
        this->state = TrackState::Deleted;
    else if ( this->time_since_update > this->_max_age )
        this->state = TrackState::Deleted;
}

bool Track::is_tentative()
{
    return this->state == TrackState::Tentative;
}

bool Track::is_confirmed()
{
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted()
{
    return this->state == TrackState::Deleted;
}
