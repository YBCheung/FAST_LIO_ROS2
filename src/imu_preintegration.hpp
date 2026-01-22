#ifndef IMU_PREINTEGRATION_HPP
#define IMU_PREINTEGRATION_HPP

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <common_lib.h>

/**
 * @brief IMU Preintegration class inspired by VINS-Fusion
 * 
 * This class implements mid-point integration for IMU measurements
 * between two keyframes, maintaining Jacobians for bias correction
 * and covariance for uncertainty propagation.
 */
class IMUPreintegration {
public:
    /**
     * @brief Constructor with initial bias
     * @param acc_bias Initial accelerometer bias
     * @param gyr_bias Initial gyroscope bias
     */
    IMUPreintegration(const V3D &acc_bias, const V3D &gyr_bias)
        : linearized_ba(acc_bias), linearized_bg(gyr_bias)
    {
        delta_p.setZero();
        delta_v.setZero();
        delta_q.setIdentity();
        
        sum_dt = 0.0;
        acc_0.setZero();
        gyr_0.setZero();
        acc_1.setZero();
        gyr_1.setZero();
        
        jacobian.setIdentity();
        covariance.setZero();
        noise.setZero();
        
        // Initialize noise matrix with default values
        // These will be set from config parameters
        double ACC_N = 0.1;      // accelerometer noise
        double GYR_N = 0.01;     // gyroscope noise
        double ACC_W = 0.0001;   // accelerometer bias random walk
        double GYR_W = 0.00001;  // gyroscope bias random walk
        
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }
    
    /**
     * @brief Set noise parameters
     * @param acc_n Accelerometer measurement noise
     * @param gyr_n Gyroscope measurement noise
     * @param acc_w Accelerometer bias random walk
     * @param gyr_w Gyroscope bias random walk
     */
    void setNoiseParams(double acc_n, double gyr_n, double acc_w, double gyr_w)
    {
        noise.setZero();
        noise.block<3, 3>(0, 0) = (acc_n * acc_n) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (gyr_n * gyr_n) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (acc_n * acc_n) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (gyr_n * gyr_n) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (acc_w * acc_w) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (gyr_w * gyr_w) * Eigen::Matrix3d::Identity();
    }
    
    /**
     * @brief Add IMU measurement for preintegration
     * @param dt Time delta
     * @param acc Accelerometer measurement
     * @param gyr Gyroscope measurement
     */
    void push_back(double dt, const V3D &acc, const V3D &gyr)
    {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        
        propagate(dt, acc, gyr);
    }
    
    /**
     * @brief Repropagate with updated bias
     * @param new_acc_bias Updated accelerometer bias
     * @param new_gyr_bias Updated gyroscope bias
     */
    void repropagate(const V3D &new_acc_bias, const V3D &new_gyr_bias)
    {
        sum_dt = 0.0;
        delta_p.setZero();
        delta_v.setZero();
        delta_q.setIdentity();
        linearized_ba = new_acc_bias;
        linearized_bg = new_gyr_bias;
        jacobian.setIdentity();
        covariance.setZero();
        
        for (size_t i = 0; i < dt_buf.size(); ++i)
        {
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
        }
    }
    
    /**
     * @brief Mid-point integration
     */
    void midPointIntegration(double _dt, 
                            const V3D &_acc_0, const V3D &_gyr_0,
                            const V3D &_acc_1, const V3D &_gyr_1,
                            const V3D &delta_p_, const Eigen::Quaterniond &delta_q_, const V3D &delta_v_,
                            const V3D &linearized_ba_, const V3D &linearized_bg_,
                            V3D &result_delta_p, Eigen::Quaterniond &result_delta_q, V3D &result_delta_v,
                            V3D &result_linearized_ba, V3D &result_linearized_bg, 
                            bool update_jacobian)
    {
        // Remove bias
        V3D un_acc_0 = delta_q_ * (_acc_0 - linearized_ba_);
        V3D un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg_;
        
        // Update rotation using mid-point gyroscope measurement
        result_delta_q = delta_q_ * deltaQ(un_gyr * _dt);
        
        // Update acceleration at mid-point
        V3D un_acc_1 = result_delta_q * (_acc_1 - linearized_ba_);
        V3D un_acc = 0.5 * (un_acc_0 + un_acc_1);
        
        // Update position and velocity
        result_delta_p = delta_p_ + delta_v_ * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v_ + un_acc * _dt;
        
        // Bias remains the same (updated separately by filter)
        result_linearized_ba = linearized_ba_;
        result_linearized_bg = linearized_bg_;
        
        if (update_jacobian)
        {
            // Update Jacobian
            V3D w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg_;
            V3D a_0_x = _acc_0 - linearized_ba_;
            V3D a_1_x = _acc_1 - linearized_ba_;
            
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;
            
            R_w_x = skewSymmetric(w_x);
            R_a_0_x = skewSymmetric(a_0_x);
            R_a_1_x = skewSymmetric(a_1_x);
            
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.setIdentity();
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q_.toRotationMatrix() * R_a_0_x * _dt * _dt + 
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * 
                                  (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity() * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * Eigen::Matrix3d::Identity() * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q_.toRotationMatrix() * R_a_0_x * _dt + 
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * 
                                  (Eigen::Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q_.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
            
            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15, 18);
            V.block<3, 3>(0, 0) = 0.25 * delta_q_.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) = 0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) = 0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity() * _dt;
            V.block<3, 3>(3, 9) = 0.5 * Eigen::Matrix3d::Identity() * _dt;
            V.block<3, 3>(6, 0) = 0.5 * delta_q_.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::Matrix3d::Identity() * _dt;
            V.block<3, 3>(12, 15) = Eigen::Matrix3d::Identity() * _dt;
            
            jacobian = F * jacobian;
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
        }
    }
    
    /**
     * @brief Evaluate residual
     */
    Eigen::Matrix<double, 15, 1> evaluate(const V3D &Pi, const Eigen::Quaterniond &Qi, const V3D &Vi, 
                                          const V3D &Bai, const V3D &Bgi,
                                          const V3D &Pj, const Eigen::Quaterniond &Qj, const V3D &Vj, 
                                          const V3D &Baj, const V3D &Bgj)
    {
        Eigen::Matrix<double, 15, 1> residuals;
        
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(0, 9);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(0, 12);
        
        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(3, 12);
        
        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(6, 9);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(6, 12);
        
        V3D dba = Bai - linearized_ba;
        V3D dbg = Bgi - linearized_bg;
        
        Eigen::Quaterniond corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg);
        V3D corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        V3D corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
        
        residuals.block<3, 1>(0, 0) = Qi.inverse() * (0.5 * G_m_s2 * V3D(0, 0, 1) * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(3, 0) = 2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(6, 0) = Qi.inverse() * (G_m_s2 * V3D(0, 0, 1) * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(9, 0) = Baj - Bai;
        residuals.block<3, 1>(12, 0) = Bgj - Bgi;
        
        return residuals;
    }
    
    // Public members for state
    V3D delta_p;                      // position preintegration
    Eigen::Quaterniond delta_q;       // rotation preintegration
    V3D delta_v;                      // velocity preintegration
    V3D linearized_ba;                // linearized acc bias
    V3D linearized_bg;                // linearized gyr bias
    
    Eigen::Matrix<double, 15, 15> jacobian;
    Eigen::Matrix<double, 15, 15> covariance;
    Eigen::Matrix<double, 18, 18> noise;
    
    double sum_dt;
    V3D acc_0, gyr_0, acc_1, gyr_1;
    
    std::vector<double> dt_buf;
    std::vector<V3D> acc_buf;
    std::vector<V3D> gyr_buf;
    
private:
    /**
     * @brief Propagate IMU measurement
     */
    void propagate(double _dt, const V3D &_acc_1, const V3D &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        
        V3D result_delta_p;
        Eigen::Quaterniond result_delta_q;
        V3D result_delta_v;
        V3D result_linearized_ba;
        V3D result_linearized_bg;
        
        midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                           linearized_ba, linearized_bg,
                           result_delta_p, result_delta_q, result_delta_v,
                           result_linearized_ba, result_linearized_bg, true);
        
        delta_p = result_delta_p;
        delta_q = result_delta_q;
        delta_v = result_delta_v;
        linearized_ba = result_linearized_ba;
        linearized_bg = result_linearized_bg;
        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
    }
    
    /**
     * @brief Create quaternion from angle-axis representation
     * @param theta Rotation vector (angle * axis)
     * @return Quaternion
     */
    Eigen::Quaterniond deltaQ(const V3D &theta)
    {
        Eigen::Quaterniond dq;
        V3D half_theta = theta;
        half_theta /= 2.0;
        dq.w() = 1.0;
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        
        double norm = half_theta.norm();
        if (norm > 1e-9)
        {
            double sin_half = sin(norm);
            double cos_half = cos(norm);
            dq.w() = cos_half;
            dq.vec() = sin_half / norm * half_theta;
        }
        
        return dq;
    }
    
    /**
     * @brief Create skew-symmetric matrix from vector
     */
    Eigen::Matrix3d skewSymmetric(const V3D &v)
    {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }
    
    double dt;
};

#endif // IMU_PREINTEGRATION_HPP
