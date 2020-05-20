#include "nn_matching.h"

Eigen::ArrayXXf _pdist( std::vector< std::vector< float > > _a, 
                        std::vector< std::vector< float > > _b )
{
    // --- Checking size of input data
    if ( !( _a.size() || _b.size() ) )
        return Eigen::ArrayXXf();
    
    // --- Converting input data to arrays
    Eigen::ArrayXXf a( int(_a.size()), int(_a.front().size()) );
    for ( size_t i = 0; i < _a.size(); i++ )
    {
        a.row(int(i)) << Eigen::Map< Eigen::ArrayXf, 0, Eigen::OuterStride<> >
                        ( _a.at(i).data(), 
                          int(_a.front().size()),
                          Eigen::OuterStride<>( int( _a.at(i).size() ) ) ).transpose();
    }
    Eigen::ArrayXXf b( int(_b.size()), int(_b.front().size()) );
    for ( size_t i = 0; i < _b.size(); i++ )
    {
        b.row(int(i)) << Eigen::Map< Eigen::ArrayXf, 0, Eigen::OuterStride<> >
                        ( _b.at(i).data(), 
                          int(_b.front().size()),
                          Eigen::OuterStride<>( int( _b.at(i).size() ) ) ).transpose();
    }
    
    // --- Sum of squared items in rows
    Eigen::ArrayXXf a2 = a.square().rowwise().sum();
    Eigen::ArrayXXf b2 = b.square().rowwise().sum();
    
    
    Eigen::ArrayXXf r2 = ( (-2.f) * (a.matrix() * b.matrix().transpose()) ).array();
    r2.matrix().colwise() += Eigen::VectorXf(a2);
    r2.matrix().rowwise() += Eigen::VectorXf(b2).transpose();
    
    // --- Checking negative values in array r2
    for (int i = 0; i < r2.size(); i++)
        if (*(r2.data() + i) < 0.f)
            *(r2.data() + i) = 0.f;//std::numeric_limits< float >::infinity();
    
    return r2;
}
Eigen::ArrayXXf _cosine_distance( std::vector< std::vector< float > > _a, 
                                  std::vector< std::vector< float > > _b, 
                                  bool data_is_normalized )
{
    // --- Checking size of input data
    if ( !( _a.size() || _b.size() ) )
        return Eigen::ArrayXXf();
    
    // --- Converting input data to arrays
    Eigen::ArrayXXf a( int(_a.size()), int(_a.front().size()) );
    for ( size_t i = 0; i < _a.size(); i++ )
    {
        a.row(int(i)) << Eigen::Map< Eigen::ArrayXf, 0, Eigen::OuterStride<> >
                        ( _a.at(i).data(), 
                          int(_a.front().size()),
                          Eigen::OuterStride<>( int( _a.at(i).size() ) ) ).transpose();
    }
    Eigen::ArrayXXf b( int(_b.size()), int(_b.front().size()) );
    for ( size_t i = 0; i < _b.size(); i++ )
    {
        b.row(int(i)) << Eigen::Map< Eigen::ArrayXf, 0, Eigen::OuterStride<> >
                        ( _b.at(i).data(), 
                          int(_b.front().size()),
                          Eigen::OuterStride<>( int( _b.at(i).size() ) ) ).transpose();
    }
    
    if ( !data_is_normalized )
    {
        auto a_norm = a.matrix().rowwise().norm();
        for ( int i = 0; i < a.rows(); i++ )
            a.row(i) /= a_norm(i);
        auto b_norm = b.matrix().rowwise().norm();
        for ( int i = 0; i < b.rows(); i++ )
            b.row(i) /= b_norm(i);
    }
    
    return 1.f - ( a.matrix() * b.matrix().transpose() ).array();
}
Eigen::ArrayXf _nn_euclidean_distance( std::vector< std::vector< float > > x, 
                                             std::vector< std::vector< float > > y )
{
    //Eigen::ArrayXf distances = _pdist(x, y).colwise().minCoeff();
    return _pdist(x, y).colwise().minCoeff();
}
Eigen::ArrayXf _nn_cosine_distance( std::vector< std::vector< float > > x, 
                                          std::vector< std::vector< float > > y )
{
    return _cosine_distance(x, y).colwise().minCoeff();
}

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric( std::string _metric,
                                                              float _matching_threshold,
                                                              unsigned _budget )
    : matching_threshold(_matching_threshold), budget(_budget)
{
    if ( (_metric == "euclidean") || (_metric == "cosine") )
    {
        metric = _metric;
        samples.clear();
    }
    else 
    {
        std::cerr << "Invalid metric; must be either 'euclidean' or 'cosine'" << std::endl;
        exit(0);
    }
}

Eigen::ArrayXf NearestNeighborDistanceMetric::operator()( std::vector< std::vector< float > > x, 
                                                           std::vector< std::vector< float > > y )
{
    if ( this->metric == "cosine" ) return _nn_cosine_distance( x, y );
    else if ( this->metric == "euclidean" ) return  _nn_euclidean_distance( x, y );
    std::cerr << "Invalid metric; must be either 'euclidean' or 'cosine'" << std::endl;
    exit(0);
}

void NearestNeighborDistanceMetric::partial_fit( std::vector< std::vector< float > > features, 
                                                 std::vector< int > targets, 
                                                 std::vector< int > active_targets )
{
    if ( features.size() != targets.size() )
    {
        std::cerr << "features and targets vector sizes do not match" << std::endl;
        exit(0);
    }
        
    std::map< int, std::deque< std::vector< float > > > samples_temp = samples;
    auto it = features.begin();
    for ( int target : targets )
    {
        samples_temp[target].push_back( *it++ );
        if ( (budget) && (samples_temp[target].size() > budget) )
            samples_temp[target].pop_front();
    }
    
    for ( int target : active_targets )
        samples[target] = samples_temp[target];
}
Eigen::ArrayXXf NearestNeighborDistanceMetric::distance( std::vector< std::vector< float > > features, 
                                                         std::vector< int > targets )
{
    Eigen::ArrayXXf cost_matrix( targets.size(), features.size() );
    
    for ( size_t i = 0; i < targets.size(); i++ )
        cost_matrix.row(int(i)) = this->operator()( std::vector< std::vector< float > >( samples.find(targets.at(i))->second.begin(), 
                                                                                         samples.find(targets.at(i))->second.end() ), 
                                                    features ).transpose();
    
    return cost_matrix;
}
