#include <fstream>
#include <iostream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

Eigen::Vector3d gravity(0,0,-9.8);

class ImuOdometry : public rclcpp::Node
{
  public:
    ImuOdometry()
    : Node("imu_odometry")
    {
      subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "/camera/imu", 10, std::bind(&ImuOdometry::topic_callback, this, std::placeholders::_1));
      initialized_ = false;

      integration_R_.setIdentity();
      integration_p_.setZero();
      integration_v_.setZero();

      current_time_ = 0;
      current_acc_.setZero();
      current_gyr_.setZero();

      last_time_ = 0;
      last_acc_.setZero();
      last_gyr_.setZero();

      // TODO:: initialize bias here
      bias_acc_.setZero();
      bias_gyr_.setZero();
      // TODO:: initialize traj_store_path_ here
      traj_store_path_ = "/home/zy/ws/inertial_ws/result/imu_odometry_test.txt";
    }

  private:
    void topic_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
    {
        // parse the msg
        double ax = imu_msg->linear_acceleration.x;
        double ay = imu_msg->linear_acceleration.y;
        double az = imu_msg->linear_acceleration.z;
        double gx = imu_msg->angular_velocity.x;
        double gy = imu_msg->angular_velocity.y;
        double gz = imu_msg->angular_velocity.z;

        current_time_ = rclcpp::Time(imu_msg->header.stamp).seconds();
        current_acc_ << ax, ay, az;
        current_gyr_ << gx, gy, gz;

        current_acc_ = current_acc_ - bias_acc_;
        current_gyr_ = current_gyr_ - bias_gyr_;

        if(initialized_)
        {
            // do integration
            double delta_t = current_time_ - last_time_;

            Eigen::Vector3d m_gyro = (current_gyr_ + last_gyr_)/2;
            Eigen::Vector3d m_accel = (current_acc_ + last_acc_)/2;

            Eigen::Matrix3d delta_R;
            delta_R = skew(m_gyro * delta_t).exp();

            integration_R_ = normalize_rotation(integration_R_ * delta_R);
            integration_p_ += 0.5 * integration_R_ * m_accel * delta_t * delta_t + integration_v_ * delta_t + 0.5 * gravity * delta_t * delta_t;
            integration_v_ += integration_R_ * m_accel * delta_t + gravity * delta_t;

            save_trajectory_to_file(integration_R_,integration_p_,current_time_);
        } else{
            // initialize
            // 1. compute integration_R_ according to acc and gravity
            Eigen::Vector3d direction_gravity = current_acc_;
            direction_gravity = direction_gravity / direction_gravity.norm();

            Eigen::Vector3d gI(0.0, 0.0, -1.0);

            Eigen::Vector3d v = gI.cross(direction_gravity); // rotation vector
            const double nv = v.norm();
            const double cosg = gI.dot(direction_gravity); // cos(theta)
            const double ang = acos(cosg); //(theta)

            Eigen::Vector3d vzg = v * ang / nv; // theta * unit rotation vector
            integration_R_ = exp_so3(vzg);

            // 2. change flag to represent initialize completed
            initialized_ = true;
        }

        last_time_ = current_time_;
        last_acc_ = current_acc_;
        last_gyr_ = current_gyr_;
    }

    Eigen::Matrix3d skew(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d v_hat;
        v_hat << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
        return v_hat;
    }

    Eigen::Matrix3d normalize_rotation(const Eigen::Matrix3d& R)
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        return svd.matrixU() * svd.matrixV().transpose();
    }

    Eigen::Matrix3d exp_so3(const Eigen::Vector3d& v)
    {
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        double x = v[0], y = v[1], z = v[2];
        const double d2 = x * x + y * y + z * z; // theta2
        const double d = sqrt(d2); // theta
        Eigen::Matrix3d W;
        W << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;
        // if (d < EPSILON)
        //     return (I + W + 0.5 * W * W);
        // else
            return (I + W * std::sin(d) / d + W * W * (1.0 - std::cos(d)) / d2);// ?
    }

    void save_trajectory_to_file(const Eigen::Matrix3d rotation, const Eigen::Vector3d translate, double time) const
    {
        static std::ofstream of_write(traj_store_path_);
        // Eigen::Matrix3d R = pose.block(0, 0, 3, 3);
        Eigen::Quaterniond q(rotation);
        of_write << std::fixed << time
                            << " " << translate(0) << " " << translate(1) << " " << translate(2)
                            << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
                            << std::endl;
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;

    bool initialized_;
    Eigen::Matrix3d integration_R_;
    Eigen::Vector3d integration_p_, integration_v_;

    double current_time_;
    Eigen::Vector3d current_acc_, current_gyr_;

    double last_time_;
    Eigen::Vector3d last_acc_, last_gyr_;

    Eigen::Vector3d bias_acc_, bias_gyr_;

    std::string traj_store_path_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuOdometry>());
    rclcpp::shutdown();

    return 0;
}