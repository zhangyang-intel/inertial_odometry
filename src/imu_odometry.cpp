#include <fstream>
#include <iostream>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>

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

      bias_acc_.setZero();
      bias_gyr_.setZero();

      traj_store_path_ = "/home/zy/ws/inertial_ws/result/imu_odometry_test.txt";
      debug_output_path_ = "/home/zy/ws/inertial_ws/result/debug.txt";
      debug_stream_.open(debug_output_path_.c_str());

      if (!debug_stream_.is_open())
      {
        std::cerr << "Failed to create debug output file " << "\n";
      }
    }

  private:
    void topic_callback(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg)
    {
        debug_stream_ << "---------------function void topic_callback() START!---------------\n";
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
            debug_stream_ << "Start to integrate!\n";
            // do integration
            double delta_t = current_time_ - last_time_;
            debug_stream_ << "Delta time is :\n" << delta_t << "\n";

            Eigen::Vector3d m_gyro = (current_gyr_ + last_gyr_)/2;
            Eigen::Vector3d m_accel = (current_acc_ + last_acc_)/2;

            Eigen::Matrix3d delta_R;
            delta_R = skew(m_gyro * delta_t).exp();

            Eigen::Vector3d transformed_acc = integration_R_ * m_accel;
            debug_stream_ << "The transformed_acc is : \n" << transformed_acc << "\n";

            Eigen::Vector3d final_acc = integration_R_ * m_accel + gravity;
            debug_stream_ << "The final_acc is : \n" << final_acc << "\n";

            integration_p_ += 0.5 * integration_R_ * m_accel * delta_t * delta_t + integration_v_ * delta_t + 0.5 * gravity * delta_t * delta_t;
            integration_v_ += integration_R_ * m_accel * delta_t + gravity * delta_t;
            integration_R_ = normalize_rotation(integration_R_ * delta_R);
            
            debug_stream_ << "integration_R_ is :\n" << integration_R_ << "\n";
            debug_stream_ << "integration_p_ is :\n" << integration_p_ << "\n";
            debug_stream_ << "integration_v_ is :\n" << integration_v_ << "\n";

            save_trajectory_to_file(integration_R_,integration_p_,current_time_);
            debug_stream_ << "Integration finished!\n";
        } else{
            // initialize
            // 1. compute integration_R_ according to acc and gravity
            debug_stream_ << "Start to initialize!\n";
            
            Eigen::Vector3d direction_gravity = -1*current_acc_;
            debug_stream_ << "The first acc is : \n" << current_acc_ << "\n";

            direction_gravity = direction_gravity / direction_gravity.norm();
            debug_stream_ << "The direction_gravity is : \n" << direction_gravity << "\n";

            Eigen::Vector3d gI(0.0, 0.0, -1.0);
            Eigen::Vector3d v = gI.cross(direction_gravity); // rotation vector
            const double nv = v.norm();

            const double cosg = gI.dot(direction_gravity); // cos(theta)
            const double ang = acos(cosg); //(theta)

            Eigen::Vector3d vzg = v * ang / nv; // theta * unit rotation vector
            integration_R_ = exp_so3(vzg).transpose();
            debug_stream_ << "The transformation from IMU frame to ENU frame is : \n" << integration_R_ << "\n";

            Eigen::Vector3d transformed_direction_gravity = integration_R_ * direction_gravity;
            debug_stream_ << "The transformed_direction_gravity is : \n" << transformed_direction_gravity << "\n";

            // The following is used to validate:
            Eigen::Vector3d transformed_acc = (-1)*integration_R_ * current_acc_;
            debug_stream_ << "The transformed_acc is : \n" << transformed_acc << "\n";

            Eigen::Vector3d residual = transformed_acc - gravity;
            debug_stream_ << "The residual between transformed_acc and gravity is : \n" << residual << "\n";

            // 2. change flag to represent initialize completed
            initialized_ = true;
            debug_stream_ << "Initialization finished!\n";
        }

        last_time_ = current_time_;
        last_acc_ = current_acc_;
        last_gyr_ = current_gyr_;
        debug_stream_ << "---------------function void topic_callback() END!---------------\n";
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
    std::string debug_output_path_;

    std::ofstream debug_stream_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuOdometry>());
    rclcpp::shutdown();

    return 0;
}