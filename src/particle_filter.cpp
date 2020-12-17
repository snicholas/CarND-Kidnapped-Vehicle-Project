/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 50;  // TODO: Set the number of particles
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.theta = theta;
    p.weight = 1.0;

    // add noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
	  double particle_y = particles[i].y;
	  double particle_theta = particles[i].theta;
	 
	  double pred_x;
	  double pred_y;
	  double pred_theta;
	  //Instead of a hard check of 0, adding a check for very low value of yaw_rate
	  if (fabs(yaw_rate) < 0.0001) {
	    pred_x = particle_x + velocity * cos(particle_theta) * delta_t;
	    pred_y = particle_y + velocity * sin(particle_theta) * delta_t;
	    pred_theta = particle_theta;
	  } else {
	    pred_x = particle_x + (velocity/yaw_rate) * (sin(particle_theta + (yaw_rate * delta_t)) - sin(particle_theta));
	    pred_y = particle_y + (velocity/yaw_rate) * (cos(particle_theta) - cos(particle_theta + (yaw_rate * delta_t)));
	    pred_theta = particle_theta + (yaw_rate * delta_t);
	  }
	  
	  normal_distribution<double> dist_x(pred_x, std_pos[0]);
	  normal_distribution<double> dist_y(pred_y, std_pos[1]);
	  normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
	  
	  particles[i].x = dist_x(gen);
	  particles[i].y = dist_y(gen);
	  particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (uint i = 0; i < observations.size(); i++) {
    
    LandmarkObs o = observations[i];
    double min_dist = 1000000;
    int map_id = -1;    
    for (uint j = 0; j < predicted.size(); j++) {
      LandmarkObs p = predicted[j];
      double cur_dist = dist(o.x, o.y, p.x, p.y);
      if (cur_dist < min_dist) {
        min_dist = cur_dist;
        map_id = p.id;
      }
    }
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < num_particles; i++) {
    double px = particles[i].x;
    double py = particles[i].y;
    double ptheta = particles[i].theta;
    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float lmx = map_landmarks.landmark_list[j].x_f;
      float lmy = map_landmarks.landmark_list[j].y_f;
      int lmid = map_landmarks.landmark_list[j].id_i;
      if(dist(px, py, lmx, lmy) < sensor_range){
        predictions.push_back(LandmarkObs{ lmid, lmx, lmy });
      }
    }
    vector<LandmarkObs> tobs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double t_x = cos(ptheta)*observations[j].x - sin(ptheta)*observations[j].y + px;
      double t_y = sin(ptheta)*observations[j].x + cos(ptheta)*observations[j].y + py;
      tobs.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
    }
    dataAssociation(predictions, tobs);
    particles[i].weight = 1.0;
    for (unsigned int j = 0; j < tobs.size(); j++) {
      double obx, oby, px, py;
      obx = tobs[j].x;
      oby = tobs[j].y;
      int associated_prediction = tobs[j].id;
      for (unsigned int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == associated_prediction) {
          px = predictions[k].x;
          py = predictions[k].y;
        }
      }
      double sx = std_landmark[0];
      double sy = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*sx*sy)) * exp( -( pow(px-obx,2)/(2*pow(sx, 2)) + (pow(py-oby,2)/(2*pow(sy, 2))) ) );
      if(obs_w==0){
        obs_w=0.0001;
      }
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }
  std::default_random_engine gen;
  std::discrete_distribution<int> dist(0, num_particles-1);
  auto index = dist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  double beta = 2*max_weight;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta = 2*max_weight;
    while (weights[index]<beta) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}