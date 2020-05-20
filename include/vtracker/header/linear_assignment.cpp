//#include "header/linear_assignment.h"

//TrackDetectId::TrackDetectId() { }

//TrackDetectId::TrackDetectId( std::vector< cv::Point2i > _matches,
//                              std::vector< int > _unmatched_tracks,
//                              std::vector< int > _unmatched_detections )
//    : matches(_matches),
//      unmatched_tracks(_unmatched_tracks),
//      unmatched_detections(_unmatched_detections) { }

//TrackDetectId min_cost_matching( Eigen::ArrayXXf distance_metric( std::vector< Track > tracks, 
//                                                                  std::vector< Detection > dets, 
//                                                                  std::vector< int > track_indices, 
//                                                                  std::vector< int > detection_indices ),
//                                 float max_distance, 
//                                 std::vector< Track > tracks, 
//                                 std::vector< Detection > detections,
//                                 std::vector< int > track_indices, 
//                                 std::vector< int > detection_indices )
//{
//    // !!!!!!!!!!!!!!!!!! Возможно добавить проверку пустых векторов, хотя они и так не будут обрабатываться
    
//    if (detection_indices.empty() || track_indices.empty())
//        return TrackDetectId( std::vector< cv::Point2i >(), 
//                              track_indices, 
//                              detection_indices );      //Nothing to match.
    
//    Eigen::ArrayXXf _cost_matrix = distance_metric( tracks, 
//                                                    detections, 
//                                                    track_indices, 
//                                                    detection_indices );
    
//    std::vector< std::vector< double > > cost_matrix;
//    for ( unsigned i = 0; i < _cost_matrix.rows(); i++ )
//    {
//        std::vector< double > cost_matrix_row;
//        for ( unsigned j = 0; j < _cost_matrix.cols(); j++ )
//        {
//            if ( _cost_matrix(i,j) > max_distance ) 
//                cost_matrix_row.push_back( double(max_distance) + 1e-5 );  // _cost_matrix(i,j) = max_distance + 1e-5f;
//            else 
//                cost_matrix_row.push_back( double(_cost_matrix(i,j)) );
//        }
//        cost_matrix.push_back( cost_matrix_row );
//    }
    
//    HungarianAlgorithm HungAlgo;
//    std::vector< int > indices;
//    std::vector< int > indices_id;
//    HungAlgo.Solve( cost_matrix, indices );
   
//    // Убираем не задействованные индексы ( индексы == -1 )
//    for ( size_t i = 0; i < indices.size(); i++ )
//        if ( indices.at(i) >= 0 )
//            indices_id.push_back( int(i) ); 
//    for ( auto it = indices.begin(); it < indices.end(); it++ )
//        if ( (*it) < 0 )
//            it = indices.erase(it);
    
    
//    TrackDetectId track_detect_id;
//    for ( size_t col = 0; col < detection_indices.size(); col++ )
//    {
//        auto result = std::find( indices.begin(), indices.end(), col );
//        if ( result == indices.end() )
//            track_detect_id.unmatched_detections.push_back( detection_indices.at(col) );
//    }
//    for ( size_t row = 0; row < track_indices.size(); row++ )
//    {
//        auto result = std::find( indices_id.begin(), indices_id.end(), row );
//        if ( result == indices.end() )
//            track_detect_id.unmatched_tracks.push_back( track_indices.at(row) );
//    }
//    for ( size_t i = 0; i < indices.size(); i++ )
//    {
//        size_t row = size_t( indices_id.at(i) );
//        size_t col = size_t( indices.at(i) );
//        int track_idx = track_indices.at( row );
//        int detection_idx = detection_indices.at( col );
//        if ( cost_matrix.at(row).at(col) > double( max_distance ) )
//        {
//            track_detect_id.unmatched_tracks.push_back( track_idx );
//            track_detect_id.unmatched_detections.push_back( detection_idx );
//        }
//        else
//        {
//            track_detect_id.matches.push_back( cv::Point2i( track_idx, detection_idx ) );
//        }
//    }
    
//    return track_detect_id;
//}

//TrackDetectId matching_cascade( Eigen::ArrayXXf distance_metric( std::vector< Track > tracks, 
//                                                                 std::vector< Detection > dets, 
//                                                                 std::vector< int > track_indices, 
//                                                                 std::vector< int > detection_indices ), 
//                                float max_distance, 
//                                int cascade_depth, 
//                                std::vector< Track > tracks, 
//                                std::vector< Detection > detections, 
//                                std::vector< int > track_indices, 
//                                std::vector< int > detection_indices )
//{
//    // !!!!!!!!!!!!!!!!!! Возможно добавить проверку пустых векторов, хотя они и так не будут обрабатываться
    
//    TrackDetectId track_detect_id;
//    track_detect_id.unmatched_detections = detection_indices;
    
//    //std::vector< int > unmatched_detections = detection_indices;
//    //std::vector< cv::Point2i > matches;
//    for ( int level = 0; level < cascade_depth; level++ )
//    {
//        if ( track_detect_id.unmatched_detections.empty() )   // No detections left
//            break;
        
//        std::vector< int > track_indices_l;
        
//        for ( int k : track_indices )
//            if ( tracks.at( size_t(k) ).time_since_update == (1 + level) )
//                track_indices_l.push_back(k);
        
//        if ( track_indices_l.empty() )        // Nothing to match at this level
//            continue;
        
//        TrackDetectId track_detect_id_temp;
        
//        track_detect_id_temp = min_cost_matching( distance_metric, 
//                                                  max_distance, 
//                                                  tracks, 
//                                                  detections, 
//                                                  track_indices_l, 
//                                                  track_detect_id.unmatched_detections );
        
//        track_detect_id.unmatched_detections = track_detect_id_temp.unmatched_detections;
        
//        for ( cv::Point2i matches_l : track_detect_id_temp.matches )
//            track_detect_id.matches.push_back( matches_l );
//    }
    
//    //std::vector< int > unmatched_tracks;
//    //unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
//    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Подумать над тем как упростить поиск не повторяющихся
//    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! элементов между двумя векторами
//    size_t j = 0;
//    for ( size_t i = 0; i < track_indices.size(); i++ )
//        if ( track_indices.at(i) < track_detect_id.matches.at(j).x )
//            track_detect_id.unmatched_tracks.push_back( track_indices.at(i) );
//        else j++;
    
//    for ( ; j < track_detect_id.matches.size(); j++ )
//        for ( auto it = track_detect_id.unmatched_tracks.begin(); it != track_detect_id.unmatched_tracks.end(); )
//            if ( (*it) == track_detect_id.matches.at(j).x )
//            {
//                it = track_detect_id.unmatched_tracks.erase(it);
//                break;
//            }
//            else ++it;
    
//    return track_detect_id;
//}

//Eigen::ArrayXXf gate_cost_matrix( KalmanFilter kf, 
//                                  Eigen::ArrayXXf cost_matrix, 
//                                  std::vector< Track > tracks,
//                                  std::vector< Detection > detections,
//                                  std::vector< int > track_indices, 
//                                  std::vector< int > detection_indices, 
//                                  float gated_cost, 
//                                  bool only_position )
//{
//    unsigned gating_dim = only_position ? 2 : 4;
    
//    float gating_threshold = chi2inv95[gating_dim];
    
//    std::vector< cv::Rect2f > measurements;
//    for ( int i : detection_indices )
//        measurements.push_back( detections.at( size_t(i) ).to_xyah() );
    
//    //Eigen::ArrayXXf cost_matrix( track_indices.size(), detection_indices.size() );
    
//    // track_indices.size() == cost_matrix.rows && gating_distance.size() == cost_matrix.cols
//    for ( size_t i = 0; i < track_indices.size(); i++ )
//    {
//        Track track = tracks.at(i);
//        std::vector< float > gating_distance = kf.gating_distance( track.mean, 
//                                                                   track.covariance,
//                                                                   measurements,
//                                                                   only_position );
//        for ( size_t j = 0; j < gating_distance.size(); j++ )
//            if ( gating_distance.at(j) > gating_threshold )
//                cost_matrix( int(i), int(j) ) = gated_cost;
//    }
    
//    return cost_matrix;
//}


