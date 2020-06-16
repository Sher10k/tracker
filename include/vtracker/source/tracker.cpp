#include "../header/tracker.h"

TrackDetectId::TrackDetectId() { }

TrackDetectId::TrackDetectId( std::vector< cv::Point2i > _matches,
                              std::vector< int > _unmatched_tracks,
                              std::vector< int > _unmatched_detections )
    : matches(_matches),
      unmatched_tracks(_unmatched_tracks),
      unmatched_detections(_unmatched_detections) { }


Tracker::Tracker( NearestNeighborDistanceMetric _metric,
                  float _max_iou_distance,
                  int _max_age, 
                  int _n_init )
    : metric(_metric), 
      max_iou_distance(_max_iou_distance), 
      max_age(_max_age), 
      n_init(_n_init),
      _next_id(1)
{
    //this->kf = KalmanFilter();
    //this->tracks.clear();
}

void Tracker::predict()
{
//    std::cout << "// --- f --- tracks.predict() \n";
    for ( Track &track : this->tracks )
        track.predict( this->kf );
}

void Tracker::update( std::vector< Detection > detections )
{
 
    //printTracker();
    //printTrack();
    // --- Run matching cascade.
    TrackDetectId mutud = this->_match( detections );
    
    // --- Update track set.
//    std::cout << "\n// --- f --- tracks.update() \n";
    for ( auto match : mutud.matches )
        this->tracks.at( size_t(match.x) ).update( this->kf,
                                                   detections.at( size_t(match.y) ) );
    
//    std::cout << "// --- f --- tracks.mark_missed() \n";
    for ( auto track_idx : mutud.unmatched_tracks )
        this->tracks.at( size_t(track_idx) ).mark_missed();
    
//    std::cout << "// --- f --- _initiate_track() \n\n";
    for ( auto detection_idx : mutud.unmatched_detections )
        this->_initiate_track( detections.at( size_t(detection_idx) ) );
    
    
    std::vector< Track > temp_tracks;
    for ( auto t : this->tracks )
        if ( not t.is_deleted() )
            temp_tracks.push_back( t );
    this->tracks = temp_tracks;
    
    // --- Update distance metric.
    std::vector< int > active_targets;
    for ( auto t : this->tracks )
        if ( t.is_confirmed() )
            active_targets.push_back( t.track_id );
    std::vector< std::vector< float > > features; 
    std::vector< int > targets;
    for ( auto &track : this->tracks )
    {
        if ( not track.is_confirmed() )
            continue;
        for ( auto feature : track.features )
            features.push_back( feature );
        for ( auto _ : track.features )
            targets.push_back( track.track_id );
        track.features.clear();
    }
    
    this->metric.partial_fit( features,
                              targets,
                              active_targets );
    
    //printTrack();
}

Eigen::ArrayXXf Tracker::gate_cost_matrix( KalmanFilter kf, 
                                           Eigen::ArrayXXf cost_matrix, 
                                           std::vector< Track > tracks,
                                           std::vector< Detection > detections,
                                           std::vector< int > track_indices, 
                                           std::vector< int > detection_indices, 
                                           float gated_cost, 
                                           bool only_position )
{
    unsigned gating_dim = only_position ? 2 : 4;
    
    float gating_threshold = chi2inv95[gating_dim];
    
    std::vector< cv::Rect2f > measurements;
    for ( int i : detection_indices )
        measurements.push_back( detections.at( size_t(i) ).to_xyah() );
//    std::cout << "// --- measurements[ " << measurements.size() << " ]: \n";
//        for ( auto measurement : measurements )
//        std::cout << " [ " << measurement.x << ", \t"
//                  << measurement.y << ", \t"
//                  << measurement.width << ", \t"
//                  << measurement.height << " ]\n";
//    std::cout << " \n";
    
    //Eigen::ArrayXXf cost_matrix( track_indices.size(), detection_indices.size() );
    
    // track_indices.size() == cost_matrix.rows && gating_distance.size() == cost_matrix.cols
    for ( size_t row = 0; row < track_indices.size(); row++ )
    {
//        std::cout << i;
        Track track = tracks.at( size_t(track_indices.at( row )) );
        std::vector< float > gating_distance = kf.gating_distance( track.mean, 
                                                                   track.covariance,
                                                                   measurements,
                                                                   only_position );
//        std::cout << "// --- " << row << " --- gating_distance[ " << gating_distance.size() << " ]: \n[ ";
//        for ( auto i : gating_distance )
//            std::cout << i << ", ";
//        std::cout << " ]\n\n";
        
        for ( size_t col = 0; col < gating_distance.size(); col++ )
            if ( gating_distance.at(col) > gating_threshold )
                cost_matrix( int(row), int(col) ) = gated_cost;
    }
//    std::cout << "// --- cost_matrix[ " << cost_matrix.rows() << " x " 
//              << cost_matrix.cols() << " ]: \n" << cost_matrix << "\n";
    return cost_matrix;
}

Eigen::ArrayXXf Tracker::gated_metric( std::vector< Track > tracks, 
                                       std::vector< Detection > detections, 
                                       std::vector< int > track_indices, 
                                       std::vector< int > detection_indices )
{
    std::vector< std::vector< float > > features;
    for ( int i : detection_indices )
        features.push_back( detections.at( size_t(i) ).feature );
//    std::cout << "// --- features[ " << features.size() << " x " << features.front().size() << " ] \n";
    
    std::vector< int > targets;
    for ( int j : track_indices )
        targets.push_back( tracks.at( size_t(j) ).track_id );
//    std::cout << "// --- targets[ " << targets.size() << " ]: \n";
//    for ( auto target : targets )
//        std::cout << target << ", ";
//    std::cout << " ]\n";
    
    Eigen::ArrayXXf cost_matrix = this->metric.distance( features, targets );
    
    cost_matrix = gate_cost_matrix( this->kf, 
                                    cost_matrix, 
                                    tracks, 
                                    detections, 
                                    track_indices, 
                                    detection_indices );
//    std::cout << "// --- cost_matrix[ " << cost_matrix.rows() << " x " 
//              << cost_matrix.cols() << " ]: \n" << cost_matrix << "\n\n";
    
    return cost_matrix;
}

Eigen::ArrayXf Tracker::iou( cv::Rect2f bbox,
                             std::vector< cv::Rect2f > candidates )
{
    size_t size_cand = candidates.size();
    
    cv::Point2f bbox_tl = bbox.tl();
    cv::Point2f bbox_br = bbox.br();
    
    Eigen::ArrayXf area_intersection = Eigen::ArrayXf::Zero( int(size_cand) );
    Eigen::ArrayXf area_candidates = Eigen::ArrayXf::Zero( int(size_cand) );
    
    for( size_t i = 0; i < size_cand; i++ )
    {
        cv::Point2f tl, br;
        tl.x = ( bbox_tl.x > candidates.at(i).tl().x ) ? bbox_tl.x : candidates.at(i).tl().x;
        tl.y = ( bbox_tl.y > candidates.at(i).tl().y ) ? bbox_tl.y : candidates.at(i).tl().y;
        br.x = ( bbox_br.x < candidates.at(i).br().x ) ? bbox_br.x : candidates.at(i).br().x;
        br.y = ( bbox_br.y < candidates.at(i).br().y ) ? bbox_br.y : candidates.at(i).br().y;
        
        float delta_x = ( br.x - tl.x );
        float delta_y = ( br.y - tl.y );
        cv::Point2f wh;
        wh.x = ( delta_x > 0.f ) ? delta_x : 0.f;
        wh.y = ( delta_y > 0.f ) ? delta_y : 0.f;
        
        area_intersection( int(i) ) = wh.x * wh.y;
        area_candidates( int(i) ) = candidates.at(i).width * candidates.at(i).height;
    }
    
    float area_bbox = bbox.width * bbox.height;
    
    return area_intersection / (area_bbox + area_candidates - area_intersection);
}

Eigen::ArrayXXf Tracker::iou_cost( std::vector< Track > tracks, 
                                   std::vector< Detection > detections, 
                                   std::vector< int > track_indices, 
                                   std::vector< int > detection_indices )
{
    // --- If there is no "indices", we take into account all the "tracks" and all the "detections".
    if ( track_indices.empty() )
        for ( int i = 0; i < int( tracks.size() ); i++ )
            track_indices.push_back( i );
    if ( detection_indices.empty() )
        for ( int j = 0; j < int( detections.size() ); j++ )
            detection_indices.push_back( j );
    
    Eigen::ArrayXXf cost_matrix = Eigen::ArrayXXf::Zero( int(track_indices.size()), 
                                                         int(detection_indices.size()) );
    
    std::vector< cv::Rect2f > candidates;
//    for ( auto detection : detections )
//        candidates.push_back( detection.tlwh );               // <-- ERROR 0.0.2
    for ( int i : detection_indices )
        candidates.push_back( detections.at( size_t(i) ).tlwh );
//    std::cout << "// --- candidates[ " << candidates.size() << " x 4 ]: \n";
//    for ( auto candidate : candidates )
//        std::cout << " [ " << candidate.x << ", \t"
//                  << candidate.y << ", \t"
//                  << candidate.width << ", \t"
//                  << candidate.height << " ]\n";
    
    
//    std::cout << "// --- bboxes: \n";
    for ( size_t i = 0; i < track_indices.size(); i++ )
    {
        if ( tracks.at( size_t(track_indices.at(i)) ).time_since_update > 1 )
        {
            for ( size_t j = 0; j < detection_indices.size(); j++ )
                cost_matrix(int(i), int(j)) = INFTY_COST;
            continue;
        }
        cv::Rect2f bbox = tracks.at( size_t(track_indices.at(i)) ).to_tlwh();
//        std::cout << i << " [ " << bbox.x << ", \t" 
//                  << bbox.y << ", \t" 
//                  << bbox.width << ", \t" 
//                  <<bbox.height << " ]\n";
        cost_matrix.row(int(i)) = 1.f - iou( bbox, candidates );
    }
    
    return cost_matrix;
}

TrackDetectId Tracker::min_cost_matching_gm( float max_distance, 
                                             std::vector< Track > tracks, 
                                             std::vector< Detection > detections,
                                             std::vector< int > track_indices, 
                                             std::vector< int > detection_indices )
{
    // !!!!!!!!!!!!!!!!!! Возможно добавить проверку пустых векторов, хотя они и так не будут обрабатываться
//    if ( track_indices.empty() )
//        for ( int i = 0; i < int( tracks.size() ); i++ )
//            track_indices.push_back( i );
//    if ( detection_indices.empty() )
//        for ( int j = 0; j < int( detections.size() ); j++ )
//            detection_indices.push_back( j );
    
    if (detection_indices.empty() || track_indices.empty())
        return TrackDetectId( std::vector< cv::Point2i >(), 
                              track_indices, 
                              detection_indices );      //Nothing to match.
    
    Eigen::ArrayXXf _cost_matrix = gated_metric( tracks, 
                                                 detections, 
                                                 track_indices, 
                                                 detection_indices );
//    std::cout << "// --- _cost_matrix[ " << _cost_matrix.rows() << " x " 
//              << _cost_matrix.cols() << " ]: \n" << _cost_matrix << "\n\n";
    
    std::vector< std::vector< double > > cost_matrix;
    for ( unsigned i = 0; i < _cost_matrix.rows(); i++ )
    {
        std::vector< double > cost_matrix_row;
        for ( unsigned j = 0; j < _cost_matrix.cols(); j++ )
        {
            if ( _cost_matrix(i,j) > max_distance ) 
                cost_matrix_row.push_back( double(max_distance) + 1e-5 );  // _cost_matrix(i,j) = max_distance + 1e-5f;
            else 
                cost_matrix_row.push_back( double(_cost_matrix(i,j)) );
        }
        cost_matrix.push_back( cost_matrix_row );
    }
//    std::cout << "// --- cost_matrix[ " << cost_matrix.size() << " x " << cost_matrix.front().size() << " ]: \n";
//    for ( auto i : cost_matrix )
//    {
//        for ( auto j : i )
//            std::cout << j << ", ";
//        std::cout << "\n";
//    }
    
    // --- Using the Hungarian algorithm, we look for the best "indices" for "cost_matrix".
    HungarianAlgorithm HungAlgo;
    std::vector< int > _indices;
    HungAlgo.Solve( cost_matrix, _indices );
        // Убираем не задействованные индексы ( индексы == -1 )
    std::vector< int > indices_id;
    std::vector< int > indices;
    for ( size_t i = 0; i < _indices.size(); i++ )
        if ( _indices.at(i) >= 0 )
        {
            indices_id.push_back( int(i) );
            indices.push_back( _indices.at(i) );
        }
//    for ( auto it = indices.begin(); it < indices.end(); it++ )
//        if ( (*it) < 0 )
//            it = indices.erase(it);                                         // --- ERROR 0.0.9
//    std::cout << "// --- indices_1[ " << indices.size() << " x 2 ]: \n";
//    for ( size_t i = 0; i < indices.size(); i++ )
//        std::cout << "[ " << indices_id.at(i) << ", " << indices.at(i) << " ]\n";
    
    
    TrackDetectId track_detect_id;
    for ( size_t col = 0; col < detection_indices.size(); col++ )
    {
        auto result = std::find( indices.begin(), indices.end(), col );
        if ( result == indices.end() )
            track_detect_id.unmatched_detections.push_back( detection_indices.at(col) );
    }
    for ( size_t row = 0; row < track_indices.size(); row++ )
    {
        auto result = std::find( indices_id.begin(), indices_id.end(), row );
        if ( result == indices_id.end() )
            track_detect_id.unmatched_tracks.push_back( track_indices.at(row) );
    }
    for ( size_t i = 0; i < indices.size(); i++ )
    {
        size_t row = size_t( indices_id.at(i) );
        size_t col = size_t( indices.at(i) );
        int track_idx = track_indices.at( row );
        int detection_idx = detection_indices.at( col );
        if ( cost_matrix.at(row).at(col) > double( max_distance ) )
        {
            track_detect_id.unmatched_tracks.push_back( track_idx );
            track_detect_id.unmatched_detections.push_back( detection_idx );
        }
        else
        {
            track_detect_id.matches.push_back( cv::Point2i( track_idx, detection_idx ) );
        }
    }
    
    return track_detect_id;
}
TrackDetectId Tracker::min_cost_matching_ic( float max_distance, 
                                             std::vector< Track > tracks, 
                                             std::vector< Detection > detections,
                                             std::vector< int > track_indices, 
                                             std::vector< int > detection_indices )
{
    // !!!!!!!!!!!!!!!!!! Возможно добавить проверку пустых векторов, хотя они и так не будут обрабатываться
//    if ( track_indices.empty() )
//        for ( int i = 0; i < int( tracks.size() ); i++ )
//            track_indices.push_back( i );
//    if ( detection_indices.empty() )
//        for ( int j = 0; j < int( detections.size() ); j++ )
//            detection_indices.push_back( j );
    
//    std::cout << "// ----------- flag_1\n";
    if (detection_indices.empty() || track_indices.empty())
        return TrackDetectId( std::vector< cv::Point2i >(), 
                              track_indices, 
                              detection_indices );      //Nothing to match.
//    std::cout << "// ----------- flag_2\n";
    
    Eigen::ArrayXXf _cost_matrix = iou_cost( tracks, 
                                             detections, 
                                             track_indices, 
                                             detection_indices );       // <-- ERROR 0.0.1
//    float temp = _cost_matrix( 16, 3 );
//    std::cout << "// --- _cost_matrix[ " << _cost_matrix.rows() << " x " 
//    << _cost_matrix.cols() << " ]: \n" << _cost_matrix << "\n\n";
    
    std::vector< std::vector< double > > cost_matrix;
    for ( unsigned i = 0; i < _cost_matrix.rows(); i++ )
    {
        std::vector< double > cost_matrix_row;
        for ( unsigned j = 0; j < _cost_matrix.cols(); j++ )
        {
            if ( _cost_matrix(i,j) > max_distance ) 
                cost_matrix_row.push_back( double(max_distance) + 1e-5 );  // _cost_matrix(i,j) = max_distance + 1e-5f;
            else 
                cost_matrix_row.push_back( double(_cost_matrix(i,j)) );
        }
        cost_matrix.push_back( cost_matrix_row );
    }
//    std::cout << "// --- cost_matrix[ " << cost_matrix.size() << " x " << cost_matrix.front().size() << " ]: \n";
//    for ( auto i : cost_matrix )
//    {
//        for ( auto j : i )
//            std::cout << j << ", ";
//        std::cout << "\n";
//    }
    
    // --- Using the Hungarian algorithm, we look for the best "indices" for "cost_matrix".
    HungarianAlgorithm HungAlgo;
    std::vector< int > _indices;
    HungAlgo.Solve( cost_matrix, _indices );
        // Убираем не задействованные индексы ( индексы == -1 )
    std::vector< int > indices_id;
    std::vector< int > indices;
    for ( size_t i = 0; i < _indices.size(); i++ )
        if ( _indices.at(i) >= 0 )
        {
            indices_id.push_back( int(i) );
            indices.push_back( _indices.at(i) );
        }
//    for ( auto it = indices.begin(); it <= indices.end(); it++ )
//        if ( (*it) < 0 )
//            it = indices.erase(it);                                         // --- ERROR 0.0.9
//    std::cout << "// --- indices_2[ " << indices.size() << " x 2 ]: \n";
//    for ( size_t i = 0; i < indices.size(); i++ )
//        std::cout << "[ " << indices_id.at(i) << ", " << indices.at(i) << " ]\n";
    
    
    TrackDetectId track_detect_id;
    for ( size_t col = 0; col < detection_indices.size(); col++ )
    {
        auto result = std::find( indices.begin(), indices.end(), col );
        if ( result == indices.end() )
            track_detect_id.unmatched_detections.push_back( detection_indices.at(col) );
    }
    for ( size_t row = 0; row < track_indices.size(); row++ )
    {
        auto result = std::find( indices_id.begin(), indices_id.end(), row );
        if ( result == indices_id.end() )
            track_detect_id.unmatched_tracks.push_back( track_indices.at(row) );
    }
    for ( size_t i = 0; i < indices.size(); i++ )
    {
        size_t row = size_t( indices_id.at(i) );
        size_t col = size_t( indices.at(i) );
        int track_idx = track_indices.at( row );
        int detection_idx = detection_indices.at( col );
        if ( cost_matrix.at(row).at(col) > double( max_distance ) )
        {
            track_detect_id.unmatched_tracks.push_back( track_idx );
            track_detect_id.unmatched_detections.push_back( detection_idx );
        }
        else
        {
            track_detect_id.matches.push_back( cv::Point2i( track_idx, detection_idx ) );
        }
    }
    
    return track_detect_id;
}

TrackDetectId Tracker::matching_cascade( float max_distance, 
                                         int cascade_depth, 
                                         std::vector< Track > tracks, 
                                         std::vector< Detection > detections, 
                                         std::vector< int > track_indices, 
                                         std::vector< int > detection_indices )
{
    // !!!!!!!!!!!!!!!!!! Возможно добавить проверку пустых векторов, хотя они и так не будут обрабатываться
    
    if ( detection_indices.empty() )
        for ( size_t i = 0; i < detections.size(); i++ )
            detection_indices.push_back( int(i) );
    
//    std::cout << "// --- track_indices: " << track_indices.size() << "\n[ ";
//    for ( int track_indice : track_indices )
//        std::cout << track_indice << ", ";
//    std::cout << " ]\n";
//    std::cout << "// --- detection_indices: " << detection_indices.size() << "\n[ ";
//    for ( int detection_indice : detection_indices )
//        std::cout << detection_indice << ", ";
//    std::cout << " ]\n";
    
    TrackDetectId track_detect_id;
    track_detect_id.unmatched_detections = detection_indices;
    
    //std::vector< int > unmatched_detections = detection_indices;
    //std::vector< cv::Point2i > matches;
    for ( int level = 0; level < cascade_depth; level++ )
    {
        if ( track_detect_id.unmatched_detections.empty() )   // No detections left
            break;
        
        std::vector< int > track_indices_l;
        
        for ( int k : track_indices )
            if ( tracks.at( size_t(k) ).time_since_update == (1 + level) )
                track_indices_l.push_back(k);
        
        if ( track_indices_l.empty() )        // Nothing to match at this level
            continue;
        
        TrackDetectId track_detect_id_l;
        
        track_detect_id_l = min_cost_matching_gm( max_distance, 
                                                  tracks, 
                                                  detections, 
                                                  track_indices_l, 
                                                  track_detect_id.unmatched_detections );
        
        track_detect_id.unmatched_detections = track_detect_id_l.unmatched_detections;
        
        for ( cv::Point2i matches_l : track_detect_id_l.matches )
            track_detect_id.matches.push_back( matches_l );
    }
    
    //std::vector< int > unmatched_tracks;
    //unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Подумать над тем как упростить поиск не повторяющихся
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! элементов между двумя векторами
//    std::cout << "// --- track_indices[ " << track_indices.size() << " ]: \n [";
//    for ( auto track_indice : track_indices )
//        std::cout << track_indice << ", ";
//    std::cout << " ]\n";
//    std::cout << "// --- matches[ " << track_detect_id.matches.size() << " ]: \n";
//    for ( auto matche : track_detect_id.matches )
//        std::cout << matche << ", ";
//    std::cout << "\n";
    
    std::vector< int > k;
    for ( auto matche : track_detect_id.matches )
        k.push_back( matche.x );
    for ( auto track_indice : track_indices )
    {
        if ( k.empty() )
        {
            track_detect_id.unmatched_tracks.push_back( track_indice );
            break;
        }
        auto result = std::find( k.begin(), k.end(), track_indice );
        if ( result == k.end() )
            track_detect_id.unmatched_tracks.push_back( track_indice );
        else
            k.erase( result );
    }
    
//    size_t j = 0;                                                                  // ERROR 0.1.0
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
//    std::cout << "// --- unmatched_tracks[ " << track_detect_id.unmatched_tracks.size() << " ]: \n";
//    for ( auto unmatched_track : track_detect_id.unmatched_tracks )
//        std::cout << unmatched_track << ", ";
//    std::cout << "\n\n";
    
    return track_detect_id;
}

TrackDetectId Tracker::_match( std::vector< Detection > & detections )
{
    // --- Split track set into confirmed and unconfirmed tracks.
//    std::cout << "\n// --- 0 --- Split track set into confirmed and unconfirmed tracks.\n";
//    std::cout << "// --- this->tracks: " << this->tracks.size() << "\n";
    std::vector< int > confirmed_tracks;
    std::vector< int > unconfirmed_tracks;
    for ( size_t i = 0; i < this->tracks.size(); i++ )
        if ( this->tracks.at(i).is_confirmed() )
            confirmed_tracks.push_back( int(i) );
        else
            unconfirmed_tracks.push_back( int(i) );
    
//    std::cout << "// --- confirmed_tracks[ " << confirmed_tracks.size() << " ]: \n [ ";
//    for ( int confirmed_track : confirmed_tracks )
//        std::cout << confirmed_track << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- unconfirmed_tracks[ " << unconfirmed_tracks.size() << " ]: \n [";
//    for ( int unconfirmed_track : unconfirmed_tracks )
//        std::cout << unconfirmed_track << ", ";
//    std::cout << " ] \n";
    
    // --- Associate confirmed tracks using appearance features.
//    std::cout << "\n // --- --- START ERROR --- --- // \n\n";
//    std::cout << "// --- 1 --- Associate confirmed tracks using appearance features.\n";
    TrackDetectId mutud_a = matching_cascade( this->metric.matching_threshold, 
                                              this->max_age,
                                              this->tracks, 
                                              detections, 
                                              confirmed_tracks );
//    std::cout << "// --- mutud_a.matches[ " << mutud_a.matches.size() << " ]: \n [ ";
//    for ( auto matche : mutud_a.matches )
//        std::cout << matche;
//    std::cout << " ] \n";
//    std::cout << "// --- mutud_a.unmatched_tracks[ " << mutud_a.unmatched_tracks.size() << " ]: \n [ ";
//    for ( int unmatched_track : mutud_a.unmatched_tracks )
//        std::cout << unmatched_track << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- mutud_a.unmatched_detections[ " << mutud_a.unmatched_detections.size() << " ]: \n [ ";
//    for ( int unmatched_detections : mutud_a.unmatched_detections )
//        std::cout << unmatched_detections << ", ";
//    std::cout << " ] \n";
//    std::cout << "\n // --- --- END ERROR --- --- // \n";
    
    // --- Associate remaining tracks together with unconfirmed tracks using IOU.
//    std::cout << "// --- 2 --- Associate remaining tracks together with unconfirmed tracks using IOU.\n";
    std::vector< int > iou_track_candidates = unconfirmed_tracks;
    std::vector< int > unmatched_tracks_a;
    for ( size_t k = 0; k < mutud_a.unmatched_tracks.size(); k++ )
        if ( this->tracks.at( size_t(mutud_a.unmatched_tracks.at(k)) ).time_since_update == 1 )
            iou_track_candidates.push_back( mutud_a.unmatched_tracks.at(k) );
        else
            unmatched_tracks_a.push_back( mutud_a.unmatched_tracks.at(k) );
    
//    std::cout << "// --- this->tracks[ " << this->tracks.size() << " ]\n";
//    std::cout << "// --- detections[ " << detections.size() << " ]\n";
//    std::cout << "// --- iou_track_candidates[ " << iou_track_candidates.size() << " ]: \n [ ";
//    for ( int iou_track_candidate : iou_track_candidates )
//        std::cout << iou_track_candidate << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- unmatched_tracks_a[ " << unmatched_tracks_a.size() << " ]: \n [ ";
//    for ( int unmatched_track_a : unmatched_tracks_a )
//        std::cout << unmatched_track_a << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- mutud_a.unmatched_detections[ " << mutud_a.unmatched_detections.size() << " ]: \n [ ";
//    for ( int unmatched_detection : mutud_a.unmatched_detections )
//        std::cout << unmatched_detection << ", ";
//    std::cout << " ] \n";
    
    TrackDetectId mutud_b = min_cost_matching_ic( this->max_iou_distance,
                                                  this->tracks,
                                                  detections,
                                                  iou_track_candidates,
                                                  mutud_a.unmatched_detections );       // <-- ERROR 0.0.0
//    std::cout << "// --- mutud_b.matches[ " << mutud_b.matches.size() << " ]: \n [ ";
//    for ( auto matche : mutud_b.matches )
//        std::cout << matche;
//    std::cout << " ] \n";
//    std::cout << "// --- mutud_b.unmatched_tracks[ " << mutud_b.unmatched_tracks.size() << " ]: \n [ ";
//    for ( int unmatched_track : mutud_b.unmatched_tracks )
//        std::cout << unmatched_track << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- mutud_b.unmatched_detections[ " << mutud_b.unmatched_detections.size() << " ]: \n [ ";
//    for ( int unmatched_detections : mutud_b.unmatched_detections )
//        std::cout << unmatched_detections << ", ";
//    std::cout << " ] \n";
    
    for ( auto matche : mutud_b.matches )
        mutud_a.matches.push_back(matche);
    
    {
//    std::vector< int > temp_un_tracks;
//    size_t size_un_track = mutud_a.unmatched_tracks.size() + mutud_b.unmatched_tracks.size();
//    std::vector< int >::iterator iter_begin_a, iter_begin_b;
//    if ( mutud_a.unmatched_tracks.size() == 0 )
//        mutud_a.unmatched_tracks = mutud_b.unmatched_tracks;
//    else if ( mutud_b.unmatched_tracks.size() )
//    {
//        for ( size_t k = 0; k < size_un_track; k++ )
//        {
//            if (iter_begin_a != mutud_a.unmatched_tracks.end())
//            {
//                if (iter_begin_b != mutud_b.unmatched_tracks.end())
//                {
//                    if ( (*iter_begin_a) < (*iter_begin_b) )
//                    {
//                        temp_un_tracks.push_back( *iter_begin_a );
//                        iter_begin_a++;
//                    }
//                    else
//                    {
//                        temp_un_tracks.push_back( *iter_begin_b );
//                        iter_begin_b++;
//                    }
//                }
//                else
//                {
//                    for ( ; iter_begin_a != mutud_a.unmatched_tracks.end(); ++iter_begin_a )
//                        temp_un_tracks.push_back( *iter_begin_a );
//                    break;
//                }
//            }
//            else
//            {
//                for ( ; iter_begin_b != mutud_b.unmatched_tracks.end(); ++iter_begin_b )
//                    temp_un_tracks.push_back( *iter_begin_b );
//                break;
//            }
//        }
//    }
//    mutud_a.unmatched_tracks = temp_un_tracks;
    }
    
    // --- Или если не нужна сортировка по возростанию
    mutud_a.unmatched_tracks = unmatched_tracks_a;
    for ( auto unmatched_track : mutud_b.unmatched_tracks )
        mutud_a.unmatched_tracks.push_back(unmatched_track);
    
    mutud_a.unmatched_detections = mutud_b.unmatched_detections;
    
    
//    std::cout << "\n// --- matches[ " << mutud_a.matches.size() << " ]: \n [ ";
//    for ( auto matche : mutud_a.matches )
//        std::cout << matche;
//    std::cout << " ] \n";
//    std::cout << "// --- unmatched_tracks[ " << mutud_a.unmatched_tracks.size() << " ]: \n [ ";
//    for ( int unmatched_track : mutud_a.unmatched_tracks )
//        std::cout << unmatched_track << ", ";
//    std::cout << " ] \n";
//    std::cout << "// --- unmatched_detections[ " << mutud_a.unmatched_detections.size() << " ]: \n [ ";
//    for ( int unmatched_detections : mutud_a.unmatched_detections )
//        std::cout << unmatched_detections << ", ";
//    std::cout << " ] \n";
    return mutud_a;
}

void Tracker::_initiate_track( Detection detection )
{
    this->kf.initiate( detection.to_xyah() );
    
    this->tracks.push_back( Track( this->kf.get_mean(), 
                                   this->kf.get_covariance(), 
                                   this->_next_id,
                                   this->n_init,
                                   this->max_age,
                                   detection.feature ) );
    
    this->_next_id++;
}

void Tracker::printTracker()
{
    std::cout << "// --- --- Tracker state --- --- //" << "\n";
    
    std::cout << "// --- metric: " << metric.metric << "\n"
              << "// matching_threshold: " << metric.matching_threshold << "\n"
              << "// budget: " << metric.budget << "\n"
              << "// samples: " << metric.samples.size() << "\n";
    for ( auto sample : metric.samples )
    {
        std::cout << "// sample[ " << sample.first << " ]: ";
        for ( auto feature : sample.second )
            std::cout<< feature.size() << "\n";
    }
    
    std::cout << "// --- max_iou_distance: " << max_iou_distance << "\n"
              << "// --- max_age: " << max_age << "\n"
              << "// --- n_init: " << n_init << "\n";
    
    std::cout << "// --- kf: " << "\n"
              << "// mean: \n" << kf.get_mean() << "\n"
              << "// covariance: \n" << kf.get_covariance() << "\n"
              << "// proj_mean: \n" << kf.get_proj_mean() << "\n"
              << "// proj_covariance: \n" << kf.get_proj_covariance() << "\n";
    
    std::cout << "// --- tracks[ " << tracks.size() << " ]: \n";
    for ( auto track : tracks )
        std::cout << " track_id: " << track.track_id << "\n"
                  << " hits: " << track.hits << "\n"
                  << " age: " << track.age << "\n"
                  << " time_since_update: " << track.time_since_update << "\n"
                  << " state: " << track.state << "\n"
                  << " features: " << track.features.size() << " \t feature: " << track.features.front().size() << "\n";
    
    std::cout << "// --- --- END state --- --- //" << "\n\n";
}
void Tracker::printTrack()
{
    std::cout << "// --- --- Tracks info --- --- //" << "\n";
    
    std::cout << "// --- tracks[ " << tracks.size() << " ]: \n";
    for ( auto track : tracks )
    {
        std::cout << "// track_id: " << track.track_id << " ---------------------- //\n";
        std::cout << " state: " << track.state << "\n";
        std::cout << " hits: " << track.hits << "\n";
        std::cout << " age: " << track.age << "\n";
        std::cout << " time_since_update: " << track.time_since_update << "\n";
//        std::cout << " features: " << track.features.size() << " \t feature: " << track.features.front().size() << "\n";
    }
    
    std::cout << "// --- --- END --- --- //" << "\n\n";
}






//std::cout << "matches: " << mutud.matches.size() << "\n";
//for ( auto match : mutud.matches )
//    std::cout << match << " ";
//std::cout << "\n";
//std::cout << "unmatched_tracks: " << mutud.unmatched_tracks.size() << "\n";
//for ( auto unmatched_track : mutud.unmatched_tracks )
//    std::cout << unmatched_track << " ";
//std::cout << "\n";
//std::cout << "unmatched_detections: " << mutud.unmatched_detections.size() << "\n";
//for ( auto unmatched_detection : mutud.unmatched_detections )
//    std::cout << unmatched_detection << " ";
//std::cout << "\n";
