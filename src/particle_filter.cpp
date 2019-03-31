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
using namespace std;

void Particle::print_particle() const {
    std::cout << "id " << id << " " << x << " " << y << " " << theta << " " << weight << std::endl;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    default_random_engine gen;
    
    num_particles = 32;  // TODO: Set the number of particles
    weights.resize(num_particles);
    
    double std_x = std[0], std_y = std[1], std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    for (int i = 0; i < num_particles; ++i) {
        Particle new_particle{};

        new_particle.id = i;
        new_particle.x = dist_x(gen);
        new_particle.y = dist_y(gen);
        new_particle.theta = dist_theta(gen);
        
        // Print your samples to the terminal.
        particles.push_back(new_particle);
    }
    is_initialized = true;
}


void Particle::prediction(double delta_t, double std_pos[],
                          double velocity, double yaw_rate) {
    //    default_random_engine gen;
    random_device rd;
    default_random_engine gen(rd());

    if (abs(yaw_rate) != 0) {
        x += velocity/yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta));
        y += velocity/yaw_rate   * (cos(theta) - cos(theta + yaw_rate * delta_t));
        theta += delta_t * yaw_rate;
    }
    else {
        x += delta_t * velocity * cos(theta);
        y += delta_t * velocity * sin(theta);
    }
    
        double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2];
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
    x = dist_x(gen);
    y = dist_y(gen);
    theta = dist_theta(gen);
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
    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    for (Particle& particle : particles) {
        particle.prediction(delta_t, std_pos, velocity, yaw_rate);
    }
}

vector<Map::single_landmark_s> ParticleFilter::dataAssociation(const Particle particle,
                                                               const double sensor_range,
                                                               const vector<Map::single_landmark_s>& predictions,
                                                               const Map &map_landmarks) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    vector<Map::single_landmark_s> candidate_landmarks{};
    for (Map::single_landmark_s single_landmark : map_landmarks.landmark_list) {
        if (dist(single_landmark.x_f, single_landmark.y_f, particle.x, particle.y) < sensor_range) {
            candidate_landmarks.push_back(single_landmark);
        }
    }
    vector<Map::single_landmark_s> associations{};
    for (Map::single_landmark_s prediction : predictions) {
        double min_dist = sensor_range * 2 + 1;
        Map::single_landmark_s target_landmark;
        for (Map::single_landmark_s candidate_landmark : candidate_landmarks) {
            double current_dist = dist(candidate_landmark.x_f, candidate_landmark.y_f, prediction.x_f, prediction.y_f);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                target_landmark = candidate_landmark;
            }
        }
        associations.push_back(target_landmark);
    }
    return associations;
}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
    // calculate normalization term
    double gauss_norm;
    gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);
    // calculate exponent
    double exponent;
    exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
    + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
    // calculate weight using normalization terms and exponent
    double weight;
    weight = gauss_norm * exp(-exponent);
    return weight;
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
    double std_obs_x = std_landmark[0], std_obs_y = std_landmark[1];
    for (int i = 0; i < particles.size(); ++i) {
        Particle& particle = particles[i];
        double x = particle.x, y = particle.y, theta = particle.theta;

        vector<Map::single_landmark_s> predictions;
        for (LandmarkObs observation : observations) {
            Map::single_landmark_s prediction;
            prediction.id_i = observation.id;
            prediction.x_f =  x + (cos(theta) * observation.x) - (sin(theta) * observation.y);
            prediction.y_f =  y + (sin(theta) * observation.x) + (cos(theta) * observation.y);
            predictions.push_back(prediction);
        }

        vector<Map::single_landmark_s> associations = dataAssociation(particle, sensor_range, predictions, map_landmarks);

        double total_prob = 1.;
        for (int i = 0; i < observations.size(); ++i) {
            Map::single_landmark_s prediction = predictions[i];
            Map::single_landmark_s association = associations[i];
            total_prob *= multiv_prob(std_obs_x, std_obs_y, prediction.x_f, prediction.y_f, association.x_f, association.y_f);
        }
        particle.weight = total_prob;
        weights[i] = particle.weight;
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    vector<Particle> new_particles (num_particles);
    random_device rd;
    default_random_engine gen(rd());
    for (int i = 0; i < num_particles; ++i) {
        discrete_distribution<int> index(weights.begin(), weights.end());
        new_particles[i] = particles[index(gen)];
        new_particles[i].id = i;
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
