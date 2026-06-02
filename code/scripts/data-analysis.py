import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import detrend


class IMUProcessor:
    def __init__(self, threshold=0.1):
        self.gravity = 9.80665  # m/s²
        self.sensor_rate = 200  # Hz
        self.dt = 1.0 / self.sensor_rate
        self.motion_threshold = threshold # Threshold for detecting stoppage (ZUPT)
        # TODO: add start and end datetime to filter data
        self.start_datetime = ""
        self.end_datetime = ""

        self.acc_cols = ['AccX(m/s²)', 'AccY(m/s²)', 'AccZ(m/s²)']
        self.velocity_cols = ['SpeedX(m/s)', 'SpeedY(m/s)', 'SpeedZ(m/s)']
        self.position_cols = ['PosX(m)', 'PosY(m)', 'PosZ(m)']
        self.gyro_cols = ['AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)']
        self.angle_cols = ['AngleX(°)', 'AngleY(°)', 'AngleZ(°)']
        self.mag_cols = ['HX(uT)', 'HY(uT)', 'HZ(uT)']
        self.quat_cols = ['Q0()', 'Q1()', 'Q2()', 'Q3()']
    
    def load_data(self, file_path, delimiter='\t'):
        try:
            self.df = pd.read_csv(file_path, delimiter=delimiter, skipinitialspace=True)

            print(f"File loaded. {len(self.df)} rows found.")
        except Exception as e:
            print(f"Error: The file {file_path} could not be imported.")
            return
    
    def clean_data(self):
        # Time correction at 200 Hz
        self.df['datetime'] = pd.to_datetime(self.df['time'])
        start_time = pd.to_datetime(self.df['datetime'].iloc[0]) # We take the first timestamp as the start
        self.df['datetime'] = start_time + pd.to_timedelta(self.df.index * self.dt, unit='s')
        self.total_samples = len(self.df)
        self.duration = (self.df['datetime'].iloc[-1] - self.df['datetime'].iloc[0]).total_seconds()

    def sensor_to_earth(self):
        quats = self.df[['Q1()', 'Q2()', 'Q3()', 'Q0()']].to_numpy()
        acc_sensor = self.df[['AccX(g)', 'AccY(g)', 'AccZ(g)']].to_numpy()

        rot = Rotation.from_quat(quats)
        acc_earth = rot.apply(acc_sensor)

        samples_calib = 2*200 # 400 samples = 2 seconds at 200 Hz
        gravity_bias_vector = np.mean(acc_earth[:samples_calib], axis=0)

        acc_net_g = acc_earth - gravity_bias_vector

        self.df['AccX(m/s²)'] = acc_net_g[:, 0] * self.gravity
        self.df['AccY(m/s²)'] = acc_net_g[:, 1] * self.gravity
        self.df['AccZ(m/s²)'] = acc_net_g[:, 2] * self.gravity
    
        self.df['SpeedX(m/s)'] = detrend(np.cumsum(self.df['AccX(m/s²)']) * self.dt)
        self.df['SpeedY(m/s)'] = detrend(np.cumsum(self.df['AccY(m/s²)']) * self.dt)
        self.df['SpeedZ(m/s)'] = detrend(np.cumsum(self.df['AccZ(m/s²)']) * self.dt)
    
        self.df['TrajectoryX(m)'] = detrend(np.cumsum(self.df['SpeedX(m/s)']) * self.dt)
        self.df['TrajectoryY(m)'] = detrend(np.cumsum(self.df['SpeedY(m/s)']) * self.dt)
        self.df['TrajectoryZ(m)'] = detrend(np.cumsum(self.df['SpeedZ(m/s)']) * self.dt)

    def plot_2d_dashboard(self):
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
        cols = ['X', 'Y', 'Z']
        colors = ['tab:red', 'tab:green', 'tab:blue']
        
        for i, axis in enumerate(cols):
            # Acceleration
            axes[i, 0].plot(self.df['datetime'], self.df[f'Acc{axis}(m/s²)'], color=colors[0], alpha=0.8)
            axes[i, 0].set_title(f"Acceleration {axis} (m/s²)")
            axes[i, 0].grid(True)
            
            # Speed
            axes[i, 1].plot(self.df['datetime'], self.df[f'Speed{axis}(m/s)'], color=colors[1], alpha=0.8)
            axes[i, 1].set_title(f"Speed {axis} (m/s)")
            axes[i, 1].grid(True)
            
            # Trajectory
            axes[i, 2].plot(self.df['datetime'], self.df[f'Trajectory{axis}(m)'], color=colors[2], alpha=0.8)
            axes[i, 2].set_title(f"Trajectory {axis} (m)")
            axes[i, 2].grid(True)
            
        plt.tight_layout()
        plt.show()

    def plot_3d_trajectory(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Data
        x = self.df['TrajectoryX(m)']
        y = self.df['TrajectoryY(m)']
        z = self.df['TrajectoryZ(m)']
        
        # Plot line
        ax.plot(x, y, z, label='Bar Path', linewidth=2, color='blue')
        
        # Start (= Green) and End (= Red) Points
        ax.scatter(x.iloc[0], y.iloc[0], z.iloc[0], color='green', s=100, label='Start', edgecolors='black')
        ax.scatter(x.iloc[-1], y.iloc[-1], z.iloc[-1], color='red', s=100, label='End', edgecolors='black')
        
        # Labels and Titles
        ax.set_xlabel('X (Sideways) [m]')
        ax.set_ylabel('Y (Front/Back) [m]')
        ax.set_zlabel('Z (Vertical) [m]')
        ax.set_title('3D Trajectory of the Barbell')
        
        # Equalize axes to prevents the movement from appearing flattened
        # Find the maximum range among the 3 axes to center the view
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend()
        plt.show()

    def acceleration_deviation_compensation(self, n_samples=100):
        calibration_data = self.df.head(n_samples)

        bias_x = calibration_data['AccX(m/s²)'].mean() - 0.0
        bias_y = calibration_data['AccY(m/s²)'].mean() - 0.0
        bias_z = calibration_data['AccZ(m/s²)'].mean() - self.gravity

        self.df['AccX(m/s²)'] = self.df['AccX(m/s²)'] - bias_x
        self.df['AccY(m/s²)'] = self.df['AccY(m/s²)'] - bias_y
        self.df['AccZ(m/s²)'] = self.df['AccZ(m/s²)'] - bias_z

        print(f"Acceleration detected Biases -> X: {bias_x:.4f}, Y: {bias_y:.4f}, Z: {bias_z:.4f}")

    def angular_velocity_deviation_compensation(self, n_samples=100):
        calibration_data = self.df.head(n_samples)

        bias_x = calibration_data['AsX(°/s)'].mean() - 0.0
        bias_y = calibration_data['AsY(°/s)'].mean() - 0.0
        bias_z = calibration_data['AsZ(°/s)'].mean() - 0.0

        self.df['AsX(°/s)'] = self.df['AsX(°/s)'] - bias_x
        self.df['AsY(°/s)'] = self.df['AsY(°/s)'] - bias_y
        self.df['AsZ(°/s)'] = self.df['AsZ(°/s)'] - bias_z

        print(f"Angular velocity detected Biases -> X: {bias_x:.4f}, Y: {bias_y:.4f}, Z: {bias_z:.4f}")

def main(file_path: str):
    processor = IMUProcessor(threshold=0.1)
    processor.load_data(file_path)
    processor.clean_data()
    processor.sensor_to_earth()
    processor.plot_2d_dashboard()
    processor.plot_3d_trajectory()

if __name__=="__main__":
    if len(sys.argv) > 1:
        sensor_file_path = sys.argv[1]
        main(sensor_file_path)
    else:
        print("Error: Please provide the path to the .txt file as an argument.")
        print("Example: python data-analysis.py /data/imu/DELPLANQUE_Nicolas/2025-10-20/20251020175724_SQUAT.txt")