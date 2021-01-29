#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;


// Helper: normalizes an angle to be in range [-M_PI : +M_PI]
static inline double normalize_angle(double angle)
{
#ifndef M_PI  
  const double M_PI = 3.14159265358979323846;
#endif

  // Constant time computation:
  if (angle < -M_PI || angle > M_PI) {
    // see https://github.com/ros/angles/blob/master/angles/include/angles/angles.h
    const double result = fmod(angle + M_PI, 2.0 * M_PI);
    if (result <= 0.0)
      return result + M_PI;
    return result - M_PI;
  }
  return angle;

  // This avoids a while loop:
  // while (angle > M_PI)  angle -= 2. * M_PI;
  // while (angle < -M_PI) angle += 2. * M_PI;
}


// HACK: Using a global pointer to a stream for logging (maybe null),
//       which is shared between all UKFs instances (see main.cpp).
extern std::ostream *s_logger;


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(std::string name) {
  name_ = name;

  // FLAG: if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // FLAG: if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd::Identity(5, 5);

  // PARAM: Tuning of process noise, see discussion in lesson 4.31.

  // Process noise standard deviation longitudinal acceleration in m/s^2
  //
  //   Vehicle (urban environment): a_max = 6 m/s²  =>  σ_a ~ 3 m/s²
  std_a_ = 3.0;     // σ_a (m/s²)

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // σ_ψdd (rad/s²)
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;
  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;
  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;
  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  Xsig_pred_.fill(0.0);

  // time when the state is true, in us
  time_us_ = 0;

  // precomputed weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; ++i) {
    weights_(i) = 0.5 / (lambda_ + n_aug_);
  }

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0.0, 0.0,
              0.0, std_radphi_ * std_radphi_, 0.0,
              0.0, 0.0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0.0,
              0.0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

// UPDATE/CORRECTION STEP: Measurement update (Posterior).
// The filter compares the "predicted" location with what the sensor measurement says.
// The predicted location and the measured location are combined to give an updated location.
// The Kalman filter will put more weight on either the predicted location or 
// the measured location depending on the uncertainty of each value.
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  // Make sure all computations are skipped if corresponding sensor is disabled
  if (!use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    return;
  }
  if (!use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    return;
  }

  // INITIALIZATION using first sensor measurement
  if (!is_initialized_)
  {
    ////////////////////////////////////////////////////////////////
    // PARAM: Initial state values for vector and covariance.
    //        See lesson 4.32 for details.
    //
    // The state vector of the CTRV model is 5 dimensional:
    //   0: px (m),
    //   1: py (m),
    //   2: |velocity| v (m/s),
    //   3: yaw angle ψ (rad)
    //   4: yaw rate ψd (rad/s)
    ////////////////////////////////////////////////////////////////
    x_.fill(0.0);
    P_ = MatrixXd::Identity(n_x_, n_x_);

    if (meas_package.sensor_type_ == meas_package.LASER) {
      double x = meas_package.raw_measurements_(0); // Lidar x (m)
      double y = meas_package.raw_measurements_(1); // Lidar y (m)
      x_ << x, y, 0.0, 0.0, 0.0;

      // std_laspx_: standard deviation x in m
      // std_laspy_: standard deviation y in m
      P_(0, 0) = std_laspx_ * std_laspx_;
      P_(1, 1) = std_laspy_ * std_laspy_;
      P_(2, 2) = std_a_ * std_a_;
      P_(3, 3) = sqrt(std_yawdd_);
    }
    else if (meas_package.sensor_type_ == meas_package.RADAR) {
      double r = meas_package.raw_measurements_(0);     // Radar r (m)
      double phi = meas_package.raw_measurements_(1);   // Radar phi (rad)
      double r_dot = meas_package.raw_measurements_(2); // Radar r_dot (m/s)
      x_ << cos(phi) * r, sin(phi) * r, r_dot, phi, 0.0;

      // std_radr_  : standard deviation radius in m
      // std_radphi_: standard deviation angle in rad
      // std_radrd_ : standard deviation radius change in m/s
      P_(0, 0) = std_radr_ * std_radr_;
      P_(1, 1) = std_radr_ * std_radr_;
      P_(2, 2) = std_radrd_ * std_radrd_;
      P_(3, 3) = std_radphi_ * std_radphi_;
    }

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    LogState(0.0, meas_package);
    return;
  }

  // Compute time step dt (s)
  double delta_t = double(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // PREDICTION STEP
  Prediction(delta_t);

  // UPDATE STEP
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

  //LogState(name_, delta_t, meas_package, nis_lidar_, nis_radar_, x_);
  LogState(delta_t, meas_package);
}

void UKF::LogState(double delta_t, const MeasurementPackage &m)
{
  static bool write_header = true;
  if (s_logger)
  {
    if (write_header) {
      *s_logger << "name,t,dt,sensor_type,"
                << "lidar_x,lidar_y,nis_lidar,"
                << "radar_r,radar_phi,radar_dr,nis_radar,"
                << "x_px,x_py,x_v,x_yaw_angle,x_yaw_rate"
                << std::endl;
      write_header = false;
    }
    *s_logger << name_ << "," << m.timestamp_ / 1E6 << "," << delta_t << ",";
    if (m.sensor_type_ == MeasurementPackage::LASER) {
      *s_logger << "lidar,"
                << m.raw_measurements_(0) << "," // x
                << m.raw_measurements_(1) << "," // y
                << nis_lidar_ << ","             // 2D lidar 95% < 5.991
                << ",,,,";                       // (no radar fields)
    }
    else {
      *s_logger << "radar,"
                << ",,,"                         // (no lidar fields)
                << m.raw_measurements_(0) << "," // r
                << m.raw_measurements_(1) << "," // phi
                << m.raw_measurements_(2) << "," // dr
                << nis_radar_ << ",";            // 3D radar 95% < 7.815
    }
    *s_logger << x_(0)<<","<<x_(1)<<","<<x_(2)<<","<<x_(3)<<","<<x_(4);
    *s_logger << std::endl;
    s_logger->flush();
  }
}

// PREDICTION STEP: State prediction after time Δt (Prior).
//
// Estimates the object's location.
// Modifies the state vector, x_. Predict sigma points, the state,
// and the state covariance matrix.
//
// The prediction step is independent of the measurement system.
void UKF::Prediction(double delta_t) {
  ////////////////////////////////////////////////////////////////
  // Generate (Augmented) Sigma Points (Lesson 4.18)
  ////////////////////////////////////////////////////////////////
  // create augmented mean vector + augmented state covariance
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented mean state of x_ + augmented covariance matrix of P_
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  ////////////////////////////////////////////////////////////////
  // Predict Sigma Points using Process Model (Lesson 4.21)
  // x_k+1 = f(x_k, nu_k)
  ////////////////////////////////////////////////////////////////
  // predict sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    // extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v   = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  ////////////////////////////////////////////////////////////////
  // Calculate mean and covariance of predicted state (Lesson 4.24)
  ////////////////////////////////////////////////////////////////
  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

// Update state based on lidar measurement (cartesian coordinate system)
//
// Uses lidar data to update the belief about the object's position.
// Calculates the lidar NIS.
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Input: 2D lidar measurements x, y
  const int n_z = 2;
  assert(meas_package.sensor_type_ == meas_package.LASER);
  assert(meas_package.raw_measurements_.size() == n_z);

  ///////////////////////////////////////////////////////////////////
  // 1) PredictLidarMeasurement
  // Use lidar data to update the belief about the object's position:

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, Xsig_pred_.cols());

  // transform sigma points into measurement space
  for (int i = 0; i < Zsig.cols(); ++i) {
    Zsig.col(i) = MapSigmaPointToLidarMeasurement(Xsig_pred_.col(i));
  }

  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S += R_lidar_;

  ///////////////////////////////////////////////////////////////////
  // 2) UpdateState
  VectorXd z_measurement = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_measurement - z_pred;

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  ///////////////////////////////////////////////////////////////////
  // 3) Compute Normalized Innovation Squares (NIS)
  nis_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

// Update state based on radar measurement (polar coordinate system)
//
// Uses radar data to update the belief about the object's position.
// Calculates the radar NIS.
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // meas_package: 2D lidar measurements x, y
  // Input: 3D radar measurement: r, phi, and r_dot
  const int n_z = 3;
  assert(meas_package.sensor_type_ == meas_package.RADAR);
  assert(meas_package.raw_measurements_.size() == n_z);

  ///////////////////////////////////////////////////////////////////
  // 1) PredictRadarMeasurement (Lesson 4.27)

  // result: mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // result: measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, Xsig_pred_.cols());
  
  // transform sigma points into measurement space
  for (int i = 0; i < Zsig.cols(); ++i) {
    Zsig.col(i) = MapSigmaPointToRadarMeasurement(Xsig_pred_.col(i));
  }
  
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // calculate innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize_angle(z_diff(1));

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  S += R_radar_;

  ///////////////////////////////////////////////////////////////////
  // 2) UpdateState (Lesson 4.30)
  VectorXd z_measurement = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < Zsig.cols(); ++i) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = normalize_angle(z_diff(1));

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = normalize_angle(x_diff(3));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z_measurement - z_pred;
  z_diff(1) = normalize_angle(z_diff(1));

  // update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S * K.transpose();

  ///////////////////////////////////////////////////////////////////
  // 3) Compute Normalized Innovation Squares (NIS)
  nis_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

inline
Eigen::VectorXd UKF::MapSigmaPointToRadarMeasurement(const VectorXd& sigmaPoint)
{
  const double p_x = sigmaPoint(0);
  const double p_y = sigmaPoint(1);
  const double v   = sigmaPoint(2);
  const double yaw = sigmaPoint(3);
  const double d = sqrt(p_x * p_x + p_y * p_y);

  VectorXd m = VectorXd(3); // r, phi, r_dot
  m << d, atan2(p_y, p_x), (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / d;
  return m;
}

inline
Eigen::VectorXd UKF::MapSigmaPointToLidarMeasurement(const VectorXd& sigmaPoint)
{
  VectorXd m = VectorXd(2); // x, y
  m << sigmaPoint(0), sigmaPoint(1);
  return m;
}