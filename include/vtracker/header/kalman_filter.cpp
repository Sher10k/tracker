#include "kalman_filter.h"

KalmanFilter::KalmanFilter()
{
    int ndim = 4;
    int dt = 1;
    
    _motion_mat = Eigen::MatrixXf::Identity( 2 * ndim, 2 * ndim );
    for (int i = 0; i < ndim; i++ ) 
        _motion_mat(i, ndim + i) = dt;
    _update_mat = Eigen::MatrixXf::Identity( ndim, 2 * ndim );
    
    _std_weight_position = 1.f / 20;
    _std_weight_velocity = 1.f / 160;
    
    mean = Eigen::MatrixXf::Zero( 1, 8 );
    covariance = Eigen::MatrixXf::Zero( 8, 8 );
    projected_mean = Eigen::MatrixXf::Zero( 1, 4 );
    projected_cov = Eigen::MatrixXf::Zero( 4, 4 );
}

void KalmanFilter::initiate( cv::Rect2f measurement )
{
    //cv::Rect mean_pos = measurement;
    mean << measurement.x, measurement.y, measurement.width, measurement.height, 0, 0, 0, 0;
    
    covariance = Eigen::MatrixXf::Zero( 8, 8 );
    covariance(0, 0) = 2.f * _std_weight_position * measurement.height;
    covariance(1, 1) = 2.f * _std_weight_position * measurement.height;
    covariance(2, 2) = 1e-2f;
    covariance(3, 3) = 2.f * _std_weight_position * measurement.height;
    covariance(4, 4) = 10.f * _std_weight_velocity * measurement.height;
    covariance(5, 5) = 10.f * _std_weight_velocity * measurement.height;
    covariance(6, 6) = 1e-5f;
    covariance(7, 7) = 10.f * _std_weight_velocity * measurement.height;
    
    //covariance = covariance.array().square();     // or 
    covariance *= covariance;
}

void KalmanFilter::predict( Eigen::Matrix<float, 1, 8 > & _mean, 
                            Eigen::Matrix<float, 8, 8 > & _covariance )
{
//    std::cout << "// --- mean_1: \n" << _mean << "\n";
//    std::cout << "// --- covariance_1: \n" << _covariance << "\n";
//    std::cout << "// --- _motion_mat: \n" << this->_motion_mat << "\n";
    // [1x8] = [1x8] * [8x8]
    this->mean = _mean * this->_motion_mat.transpose();     // need to rethink
    
    Eigen::Matrix<float, 8, 8 > motion_cov = Eigen::MatrixXf::Zero( 8, 8 );
    motion_cov(0,0) = _std_weight_position * _mean(3);
    motion_cov(1,1) = _std_weight_position * _mean(3);
    motion_cov(2,2) = 1e-2f;
    motion_cov(3,3) = _std_weight_position * _mean(3);
    motion_cov(4,4) = _std_weight_velocity * _mean(3);
    motion_cov(5,5) = _std_weight_velocity * _mean(3);
    motion_cov(6,6) = 1e-5f;
    motion_cov(7,7) = _std_weight_velocity * _mean(3);
    //motion_cov = motion_cov.array().square();     // or 
    motion_cov *= motion_cov;
    
    // [8x8] = ( [8x8] * [8x8] * [8x8].T ) + [8x8]
    this->covariance = (this->_motion_mat * _covariance * this->_motion_mat.transpose()) + motion_cov;
    
//    std::cout << "// --- mean_2: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_2: \n" << this->covariance << "\n\n";
}

void KalmanFilter::project( Eigen::Matrix< float, 1, 8 > & _mean,
                            Eigen::Matrix< float, 8, 8 > & _covariance )
{
//    std::cout << "// --- f --- tracks.kf.project() \n";
//    std::cout << "// --- mean_1: \n" << _mean << "\n";
//    std::cout << "// --- covariance_1: \n" << _covariance << "\n";
    // [1x4] = ( [4x8] * [1x8].T ).T
    this->projected_mean = (this->_update_mat * _mean.transpose()).transpose();
    
    Eigen::Matrix< float, 4, 4 > innovation_cov = Eigen::Matrix< float, 4, 4 >::Zero( 4, 4 );
    innovation_cov(0,0) = _std_weight_position * _mean(3);
    innovation_cov(1,1) = _std_weight_position * _mean(3);
    innovation_cov(2,2) = 1e-1f;
    innovation_cov(3,3) = _std_weight_position * _mean(3);
    //innovation_cov = innovation_cov.array().square();     // or
    innovation_cov *= innovation_cov;
    
    // [4x4] = ( [4x8] * [8x8] * [4x8].T ) + [4x4]
    this->projected_cov = (this->_update_mat * _covariance * this->_update_mat.transpose()) + innovation_cov; // [4x4]
//    std::cout << "// --- projected_mean_2: \n" << this->projected_mean << "\n";
//    std::cout << "// --- projected_covariance_2: \n" << this->projected_cov << "\n";
}

void KalmanFilter::update( Eigen::Matrix< float, 1, 8 > & _mean,
                           Eigen::Matrix< float, 8, 8 > & _covariance,
                           cv::Rect2f _measurement )
{
    this->project( _mean, _covariance );
    
    Eigen::LLT< Eigen::Matrix< float, 4, 4 > > lltofA( this->projected_cov );
    Eigen::Matrix< float, 4, 4 > chol_factor = lltofA.matrixL();
//    std::cout << "// --- chol_factor[ " << chol_factor.rows() << " x " 
//              << chol_factor.cols() << " ]: \n" << chol_factor << "\n";
    
    // [8x4] = lltofA.solve( ([8x8] * [4x8].T).T ).T
    Eigen::Matrix< float, 8, 4 > kalman_gain = lltofA.solve( ( _covariance * 
                                                               this->_update_mat.transpose() ).transpose() ).transpose();
//    std::cout << "// --- kalman_gain[ " << kalman_gain.rows() << " x " 
//              << kalman_gain.cols() << " ]: \n" << kalman_gain << "\n";
    
//    std::cout << "// --- _measurement: \n" << Eigen::Matrix< float, 1, 4 >( _measurement.x, 
//                                                                          _measurement.y, 
//                                                                          _measurement.width, 
//                                                                          _measurement.height ) << "\n";
    // convert Rect to Matrix
    Eigen::Matrix< float, 1, 4 > innovation = Eigen::Matrix< float, 1, 4 >( _measurement.x, 
                                                                            _measurement.y, 
                                                                            _measurement.width, 
                                                                            _measurement.height ) - this->projected_mean;
//    std::cout << "// --- innovation[ " << innovation.rows() << " x " 
//              << innovation.cols() << " ]: \n" << innovation << "\n";
    
    // [1x8] = [1x8] + ( [1x4] * [8x4].T )
    //this->mean += innovation * kalman_gain.transpose();               // --- ERROR 0.0.6
    this->mean = _mean + ( innovation * kalman_gain.transpose() );
    // [8x8] = [8x8] - ( [8x4] * [4x4] * [8x4].T )
    //this->covariance -= ( kalman_gain * projected_cov * kalman_gain.transpose() );    // --- ERROR 0.0.6
    this->covariance = _covariance - ( kalman_gain * projected_cov * kalman_gain.transpose() );
//    std::cout << "// --- mean_2: \n" << this->mean << "\n";
//    std::cout << "// --- covariance_2: \n" << this->covariance << "\n\n";
}

std::vector< float > KalmanFilter::gating_distance( Eigen::Matrix< float, 1, 8 > & _mean,
                                                    Eigen::Matrix< float, 8, 8 > & _covariance,
                                                    std::vector< cv::Rect2f > _measurements,
                                                    bool only_position = true )
{
    this->project( _mean, _covariance );
    
    Eigen::LLT< Eigen::Matrix< float, 4, 4 > > lltofA( this->projected_cov );
    Eigen::Matrix< float, 4, 4 > cholesky_factor = lltofA.matrixL();
//    std::cout << "// --- cholesky_factor[ " << cholesky_factor.rows() << " x " 
//              << cholesky_factor.cols() << " ]: \n" << cholesky_factor << "\n";
    
    Eigen::MatrixXf d( _measurements.size(), 4 );
    for ( size_t i = 0; i < _measurements.size(); i++ )
    {
        d( int(i), 0 ) = _measurements.at(i).x - this->projected_mean(0);
        d( int(i), 1 ) = _measurements.at(i).y - this->projected_mean(1);
        d( int(i), 2 ) = _measurements.at(i).width - this->projected_mean(2);
        d( int(i), 3 ) = _measurements.at(i).height - this->projected_mean(3);
    }
//    std::cout << "// --- d[ " << d.rows() << " x " << d.cols() << " ]: \n" << d << "\n";
    
    Eigen::MatrixXf z( 4, _measurements.size() );
    Eigen::LLT< Eigen::Matrix< float, 4, 4 > > lltofA_z( cholesky_factor );
//    z = lltofA.solve( d.transpose() );                    // --- ERROR 0.0.7
    z = lltofA_z.solve( d.transpose() );
//    std::cout << "// --- z[ " << z.rows() << " x " << z.cols() << " ]: \n" << z << "\n";
    
//    z.array().square();                           // --- ERROR 0.0.8
    z = z.array().square();
//    std::cout << "// --- z[ " << z.rows() << " x " << z.cols() << " ]: \n" << z << "\n";
    std::vector< float > squared_maha;
    for ( size_t i = 0; i < _measurements.size(); i++ )
        squared_maha.push_back( z(0, int(i)) + 
                                z(1, int(i)) + 
                                z(2, int(i)) + 
                                z(3, int(i)) );
//    std::cout << "// --- squared_maha[ " << squared_maha.size() << " ]: \n[ ";
//    for ( auto i : squared_maha )
//        std::cout << i << ", ";
//    std::cout << " ]\n\n";
    return squared_maha;
}

Eigen::MatrixXf KalmanFilter::get_mean()
{
    return this->mean;
}
Eigen::MatrixXf KalmanFilter::get_covariance()
{
    return this->covariance;
}
Eigen::MatrixXf KalmanFilter::get_proj_mean()
{
    return this->projected_mean;
}
Eigen::MatrixXf KalmanFilter::get_proj_covariance()
{
    return this->projected_cov;
}
