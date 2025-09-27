
import json
import numpy as np
from csv import reader
from utils.logger import setup_logger


class TelemetryImporter:
    ''' TelemetryImporter

    A class responsible for importing telemetry data from various sources, including GoPro JSON files, CSV files, and other JSON formats.
    It processes the telemetry data, applies necessary transformations, and stores the results in a structured format.
    '''
    def __init__(self, logger=None):    
        self.ms_to_sec = 1e-3
        self.us_to_sec = 1e-6
        self.ns_to_sec = 1e-9

        self.telemetry = {}

        if logger is None:
            self.logger = setup_logger(prefix="TelemetryImporter")
        else:
            self.logger = logger

    def _remove_seconds(self, accl, gyro, timestamps_ns, skip_seconds):
        ''' 
        Remove data points from the beginning and end of the telemetry data based on the specified number of seconds to skip.

        Parameters:
        accl (list): Accelerometer data.
        gyro (list): Gyroscope data.
        timestamps_ns (list): Timestamps in nanoseconds.
        skip_seconds (float): Number of seconds to skip from the beginning and end.

        Returns:
        tuple: Processed accelerometer, gyroscope, and timestamp data.
        '''
        skip_ns = skip_seconds / self.ns_to_sec

        ds = timestamps_ns[1] - timestamps_ns[0]
        nr_remove = round(skip_ns / ds)

        accl = accl[nr_remove:len(timestamps_ns) - nr_remove]
        gyro = gyro[nr_remove:len(timestamps_ns) - nr_remove]

        timestamps_ns = timestamps_ns[nr_remove:len(timestamps_ns) - nr_remove]

        return accl, gyro, timestamps_ns

    def read_gopro_telemetry(self, path_to_jsons, skip_seconds=0.0):
        '''
        Read telemetry data from GoPro JSON files.

        Parameters:
        path_to_jsons (str or list): Path to a single JSON file or a list of paths for multiple files.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        
        if isinstance(path_to_jsons, (list, tuple)):
            accl = []
            gyro = []
            timestamps_ns = []
            image_timestamps_ns = []
            last_timestamp, last_img_timestamp = 0.0, 0.0

            for p in path_to_jsons:
                telemetry = self._read_gopro_telemetry(p, skip_seconds=0.0)
                accl.extend(telemetry["accelerometer"])
                gyro.extend(telemetry["gyroscope"])
                times = last_timestamp + np.asarray(telemetry["timestamps_ns"])
                img_times = last_img_timestamp + np.asarray(telemetry["img_timestamps_ns"])

                last_img_timestamp = img_times[-1]
                last_timestamp = times[-1]
                print("Setting last sensor time to: ",last_timestamp)
                print("Setting last image time to: ",last_img_timestamp)

                timestamps_ns.extend(times.tolist())
                image_timestamps_ns.extend(img_times.tolist())

            if skip_seconds != 0.0:
                accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)
                accl = accl[0:len(timestamps_ns)]
                gyro = gyro[0:len(timestamps_ns)]
            
            self.telemetry["accelerometer"] = accl
            self.telemetry["gyroscope"] = gyro
            self.telemetry["timestamps_ns"] = timestamps_ns
            self.telemetry["img_timestamps_ns"] = image_timestamps_ns
            self.telemetry["camera_fps"] = telemetry["camera_fps"]
        else:
            self.telemetry = self._read_gopro_telemetry(path_to_jsons, skip_seconds=skip_seconds)


    def _read_gopro_telemetry(self, path_to_json, skip_seconds=0.0):
        '''
        Read telemetry data from a single GoPro JSON file.

        Parameters:
        path_to_json (str): Path to the JSON file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.

        Returns:
        dict: Processed telemetry data.
        '''
        with open(path_to_json, 'r') as f:
            json_data = json.load(f)

        accl, gyro, cori, gravity  = [], [], [], []
        timestamps_ns, cori_timestamps_ns, gps_timestamps_ns = [], [], []
        gps_llh, gps_prec, gps_vel3d = [], [], []

        for a in json_data['1']['streams']['ACCL']['samples']:
            timestamps_ns.append(a['cts'] * self.ms_to_sec / self.ns_to_sec)
            accl.append([a['value'][1], a['value'][2], a['value'][0]])
        for g in json_data['1']['streams']['GYRO']['samples']:
            gyro.append([g['value'][1], g['value'][2], g['value'][0]])
        # image orientation at framerate
        for c in json_data['1']['streams']['CORI']['samples']:
            # order w,x,z,y https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
            w, x, z, y = c['value'][0], c['value'][1], c['value'][2], c['value'][3]
            cori.append([x, y, z, w])
            cori_timestamps_ns.append(c['cts'] * self.ms_to_sec / self.ns_to_sec)
        
        # gravity vector in camera coordinates at framerate
        for g in json_data['1']['streams']['GRAV']['samples']:
            # https://github.com/gopro/gpmf-parser/issues/170 x, -z, -y
            # gravity vector in camera coordinates at framerate
            gravity.append([g['value'][0], g['value'][2], g['value'][1]])
            #gravity.append([g['value'][0], g['value'][1], g['value'][2]])
        # GPS
    
        for g in json_data["1"]["streams"]["GPS5"]["samples"]:
            if g["fix"] != 0:
                gps_timestamps_ns.append(g['cts'] * self.ms_to_sec / self.ns_to_sec)
                lat, long, alt = g["value"][0], g["value"][1], g["value"][2]
                gps_llh.append([lat,long,alt])
                gps_prec.append(g["precision"])
                gps_vel3d.append(g["value"][4])

        camera_fps = json_data['frames/second']
        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = camera_fps
        self.telemetry["gravity"] = gravity 
        self.telemetry["camera_orientation"] = cori
        self.telemetry["img_timestamps_ns"] = cori_timestamps_ns

        self.telemetry["gps_llh"] = gps_llh
        self.telemetry["gps_precision"] = gps_prec
        self.telemetry["gps_timestamps_ns"] = gps_timestamps_ns
        self.telemetry["gps_vel3d"] = gps_vel3d
        return self.telemetry

    def read_csv(self, path_to_csv, skip_seconds=0.0):
        '''
        Read telemetry data from a CSV file.

        Parameters:
        path_to_csv (str): Path to the CSV file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        accl = []
        gyro  = []
        timestamps_ns = []

        # open file in read mode
        with open(path_to_csv, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                accl.append([float(row[4]),float(row[5]),float(row[6])])
                gyro.append([float(row[1]),float(row[2]),float(row[3])])
                timestamps_ns.append(float(row[0]))
        # our timestamps should always start at zero for the camera, so we normalize here

        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = 0.0
        self.telemetry["img_timestamps_ns"] = []

    def read_generic_json(self, path_to_json, skip_seconds=0.0):
        '''
        Read telemetry data from a generic JSON file.

        Parameters:
        path_to_json (str): Path to the JSON file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        with open(path_to_json, 'r') as f:
            json_data = json.load(f)

        for key in json_data:
            self.telemetry[key] = json_data[key]

    def read_zed_jsonl(self, path_to_jsonl, skip_seconds=0.0):
        '''
        Read telemetry data from a ZED JSON Lines file.

        Parameters:
        path_to_jsonl (str): Path to the JSON Lines file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        with open(path_to_jsonl, 'r') as f:
            json_list = [json.loads(line) for line in f]

        accl = []
        gyro  = []
        imu_timestamps_s = []
        frametimes_s = []
        for json_str in json_list:
            if "sensor" in json_str:
                if json_str["sensor"]["type"] == "gyroscope":
                    gyro.append(json_str["sensor"]["values"])
                    imu_timestamps_s.append(json_str["time"])
                elif json_str["sensor"]["type"] == "accelerometer":
                    accl.append(json_str["sensor"]["values"])
                
            elif "frames" in json_str:
                frametimes_s.append(json_str["time"])
        # now get imu samples inside camera times
        imu_timestamps_s = np.array(imu_timestamps_s)
        frametimes_s = np.array(frametimes_s)
        gyro = np.array(gyro)
        accl = np.array(accl)
        # get only IMU data in the frametime range
        imu_ids = np.equal(
            imu_timestamps_s >= frametimes_s[0], imu_timestamps_s <= frametimes_s[-1])

        gyro = gyro[imu_ids,:]
        accl = accl[imu_ids,:]
        imu_timestamps_ns = imu_timestamps_s[imu_ids] / self.ns_to_sec
        imu_timestamps_ns -= imu_timestamps_ns[0]
        
        if skip_seconds != 0.0:
            accl, gyro, imu_timestamps_ns = self._remove_seconds(accl, gyro, imu_timestamps_ns, skip_seconds)

        accl = accl[0:len(imu_timestamps_ns)]
        gyro = gyro[0:len(imu_timestamps_ns)]

        frametimes_s = np.array(frametimes_s)
        self.telemetry["accelerometer"] = accl.tolist()
        self.telemetry["gyroscope"] = gyro.tolist()
        self.telemetry["timestamps_ns"] = imu_timestamps_ns.tolist()
        self.telemetry["camera_fps"] = 1/np.mean(np.array(frametimes_s[1:] - frametimes_s[:-1]))
        self.telemetry["img_timestamps_ns"] = []

    def read_pygpmf_json(self, path_to_json, skip_seconds=0.0):
        '''
        Read telemetry data from a Pygpmf JSON file.

        Parameters:
        path_to_json (str): Path to the JSON file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        with open(path_to_json, 'r') as f:
            json_data = json.load(f)

        accl, gyro = [], []
        timestamps_ns = []
        img_timestamps_ns = []
        gps_timestamps_ns = []

        for a in json_data['ACCL']['data']:
            accl.append([a[1], a[2], a[0]])
        for g in json_data['GYRO']['data']:
            gyro.append([g[1], g[2], g[0]])
        for t in json_data['ACCL']['timestamps_s']:
            timestamps_ns.append(t/self.ns_to_sec)
        for t in json_data['img_timestamps_s']:
            img_timestamps_ns.append(t/self.ns_to_sec)
        
        # image orientation at framerate
        if "CORI" in json_data:
            cori = []
            for c in json_data['CORI']['data']:
                # order w,x,z,y https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
                w, x, z, y = c[0], c[1], c[2], c[3]
                cori.append([x, y, z, w])
            self.telemetry["camera_orientation"] = cori
            
        # image orientation at framerate
        if "IORI" in json_data:
            iori = []
            for c in json_data['IORI']['data']:
                # order w,x,z,y https://github.com/gopro/gpmf-parser/issues/100#issuecomment-656154136
                w, x, z, y = c[0], c[1], c[2], c[3]
                iori.append([x, y, z, w])
            self.telemetry["image_orientation"] = iori
            
        if "GRAV" in json_data:
            # gravity vector in camera coordinates at framerate
            self.telemetry["gravity"] = json_data['GRAV']['data'] 
            # switch to x z y
            # grav = np.array(self.telemetry["gravity"])[:,[0,2,1]]
            # self.telemetry["gravity"] = grav.tolist()
            self.telemetry["gravity_timestamps_ns"] = (
                np.array(json_data['GRAV']['timestamps_s']) / self.ns_to_sec).tolist()
            
        # GPS is optional
        if "GPS5" in json_data:
            self.telemetry["gps_llh"] = json_data["GPS5"]["data"]
            for t in json_data["GPS5"]["timestamps_s"]:
                gps_timestamps_ns.append(t/self.ns_to_sec)
            self.telemetry["gps_timestamps_ns"] = gps_timestamps_ns

            if "GPSF" in json_data:
                gps_fix_times_ns = np.array(json_data["GPSF"]["timestamps_s"])/self.ns_to_sec
                gps_fix_data = np.array(json_data["GPSF"]["data"])
                gps_fix_sticky = np.zeros(len(gps_timestamps_ns))
                for i in range(len(gps_fix_sticky)):
                    gps_fix_sticky[i] = gps_fix_data[np.where(gps_fix_times_ns <= gps_timestamps_ns[i])[0][-1]]

                self.telemetry["gps_fix"] = gps_fix_sticky.tolist()

            if "GPSP" in json_data:
                gps_prec_times_ns = np.array(json_data["GPSP"]["timestamps_s"])/self.ns_to_sec
                gps_prec_data = np.array(json_data["GPSP"]["data"])
                gps_prec_sticky = np.zeros(len(gps_timestamps_ns))
                for i in range(len(gps_prec_sticky)):
                    gps_prec_sticky[i] = gps_prec_data[np.where(gps_prec_times_ns <= gps_timestamps_ns[i])[0][-1]]

                self.telemetry["gps_precision"] = gps_prec_sticky.tolist()

        camera_fps = 1 /  (np.mean(np.diff(img_timestamps_ns))*self.ns_to_sec)
        if skip_seconds != 0.0:
            accl, gyro, timestamps_ns = self._remove_seconds(accl, gyro, timestamps_ns, skip_seconds)

        accl = accl[0:len(timestamps_ns)]
        gyro = gyro[0:len(timestamps_ns)]

        self.telemetry["accelerometer"] = accl
        self.telemetry["gyroscope"] = gyro
        self.telemetry["timestamps_ns"] = timestamps_ns
        self.telemetry["camera_fps"] = camera_fps
        self.telemetry["img_timestamps_ns"] = img_timestamps_ns

    def get_gps_pos_at_frametimes(self, img_times_ns=None):
        '''
        Interpolate GPS coordinates for each frame based on available telemetry data.

        Parameters:
        img_times_ns (list): List of image timestamps in nanoseconds.

        Returns:
        tuple: Interpolated GPS coordinates, GPS precision, camera velocity, and interpolated frame times.
        '''
        import pymap3d
        # interpolate camera gps info at frametimes
        frame_gps_ecef = [
            pymap3d.geodetic2ecef(llh[0],llh[1],llh[2]) for llh in self.telemetry["gps_llh"]]
        frame_gps_ecef = np.array(frame_gps_ecef)
        gps_times = np.array(self.telemetry["gps_timestamps_ns"]) * self.ns_to_sec
        if img_times_ns is not None:
            frame_times = np.array(img_times_ns) * self.ns_to_sec
        else:
            frame_times = np.array(self.telemetry["img_timestamps_ns"]) * self.ns_to_sec
        
        # find valid interval (interpolate only where we actually have gps measurements)
        start_frame_time_idx = np.where(gps_times[0] < frame_times)[0][0]
        if gps_times[-1] > frame_times[-1]:
            end_frame_time_idx = len(frame_times)
        else:
            end_frame_time_idx = np.where(gps_times[-1] <= frame_times)[0][0]
            if not end_frame_time_idx:
                end_frame_time_idx = len(frame_times)

        cam_hz = 1 / self.telemetry["camera_fps"]
        if img_times_ns is not None:
            interp_frame_times = frame_times[start_frame_time_idx:end_frame_time_idx]
        else:
            interp_frame_times = np.round(
            np.arange(
                np.round(frame_times[start_frame_time_idx],2), 
                np.round(frame_times[end_frame_time_idx],2) - cam_hz, cam_hz) ,3).tolist()

        x_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,0])
        y_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,1])
        z_interp = np.interp(interp_frame_times, gps_times, frame_gps_ecef[:,2])
        prec_interp = np.interp(interp_frame_times, gps_times, self.telemetry["gps_precision"])

        xyz_interp = np.stack([x_interp,y_interp,z_interp],1)

        interp_frame_times_ns = (np.array(interp_frame_times)*1e9).astype(np.int64)
        gps_prec = dict(zip(interp_frame_times_ns, prec_interp.tolist()))

        camera_gps = dict(zip(interp_frame_times_ns, xyz_interp.tolist()))

        if "gps_vel3d" in self.telemetry:
            vel_interp = np.interp(interp_frame_times, gps_times, self.telemetry["gps_vel3d"])
            camera_vel3d = dict(zip(interp_frame_times_ns, vel_interp.tolist()))
        else:
            camera_vel3d = None
        
        return camera_gps, gps_prec, camera_vel3d, interp_frame_times_ns

    def get_gravity_at_times(self, interp_times_ns, R_c_i=np.eye(3)):
        '''
        Interpolate gravity vector at specified times.

        Parameters:
        interp_times_ns (list): List of interpolation times in nanoseconds.
        R_c_i (numpy.ndarray): Rotation matrix from camera to inertial frame.

        Returns:
        dict: Interpolated gravity vector at specified times.
        '''
        if interp_times_ns is not None:
            interp_times_s = np.array(interp_times_ns) * self.ns_to_sec
        else:
            interp_times_s = np.array(self.telemetry["img_timestamps_ns"]) * self.ns_to_sec

        grav_vector = np.array(self.telemetry["gravity"])
        grav_times_s= np.array(self.telemetry["gravity_timestamps_ns"]) * self.ns_to_sec
        # find valid interval (interpolate only where we actually have measurements)
        start_frame_time_idx = np.where(grav_times_s[0] < interp_times_s)[0][0]
        if grav_times_s[-1] > interp_times_s[-1]:
            end_frame_time_idx = len(interp_times_s)-1
        else:
            end_frame_time_idx = np.where(grav_times_s[-1] <= interp_times_s)[0][0]
            if not end_frame_time_idx:
                end_frame_time_idx = len(interp_times_s)-1

        interp_frame_times = interp_times_s[start_frame_time_idx:end_frame_time_idx]

        x_interp = np.interp(interp_frame_times, grav_times_s, grav_vector[:,0])
        y_interp = np.interp(interp_frame_times, grav_times_s, grav_vector[:,1])
        z_interp = np.interp(interp_frame_times, grav_times_s, grav_vector[:,2])
        frame_grav_interp = (R_c_i @ np.stack([x_interp,y_interp,z_interp],1).T).T
        frame_grav_interp_dict = dict(zip((np.array(interp_frame_times)*1e9).astype(np.int64), 
                                          frame_grav_interp.tolist()))
        return frame_grav_interp_dict

class TelemetryConverter:
    ''' TelemetryConverter

    A class responsible for converting telemetry data from various formats into a structured output.
    It utilizes the TelemetryImporter class to read and process telemetry data, and provides methods to export the data
    in different formats, such as JSON and CSV.
    '''
    def __init__(self, logger=None):
        self.output_dict = {}
        self.telemetry_importer = TelemetryImporter()

        self.logger = setup_logger(prefix="TelemetryConverter") if logger is None else logger

    def _dump_final_json(self, output_path):
        '''
        Dump the processed telemetry data to a JSON file.

        Parameters:
        output_path (str): Path to the output JSON file.
        '''
        self.logger.info(f"Dumping telemetry to {output_path}")
        with open(output_path, "w") as f:
            json.dump(self.telemetry_importer.telemetry, f)

    def _dump_kalibr_csv(self, output_path):
        '''
        Dump the processed telemetry data to a Kalibr CSV file.

        Parameters:
        output_path (str): Path to the output CSV file.
        '''
        self.logger.info(f"Dumping telemetry to {output_path}")
        from utils import time_to_s_nsec
        with open(output_path, "w") as f:
            for i in range(len(self.telemetry_importer.telemetry["timestamps_ns"])):
                t = self.telemetry_importer.telemetry["timestamps_ns"][i]
                g = self.telemetry_importer.telemetry["gyroscope"][i]
                a = self.telemetry_importer.telemetry["accelerometer"][i]
                t_s, t_ns = time_to_s_nsec(t*1e-9)
                f.write(str(t_s)+format(int(t_ns), '09d')+','+str(g[0])+','+str(g[1])+','+str(g[2])+','+str(a[0])+','+str(a[1])+','+str(a[2])+'\n')
            f.close()

    def convert_gopro_telemetry_file(self, input_telemetry_json, output_path, skip_seconds=0.0):
        '''
        Convert a GoPro telemetry JSON file to a structured output format.

        Parameters:
        input_telemetry_json (str): Path to the input telemetry JSON file.
        output_path (str): Path to the output file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        self.logger.info(f"Converting gopro telemetry file {input_telemetry_json} to {output_path}")
        self.telemetry_importer.read_gopro_telemetry(
            input_telemetry_json, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)

    def convert_gopro_telemetry_file_to_kalibr(self, input_telemetry_json, output_path, skip_seconds=0.0):
        '''
        Convert a GoPro telemetry JSON file to a Kalibr CSV format.

        Parameters:
        input_telemetry_json (str): Path to the input telemetry JSON file.
        output_path (str): Path to the output CSV file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        self.logger.info(f"Converting gopro telemetry file {input_telemetry_json} to {output_path}")
        self.telemetry_importer.read_gopro_telemetry(
            input_telemetry_json, skip_seconds=skip_seconds)
        self._dump_kalibr_csv(output_path)

    def convert_csv_telemetry_file(self, csv_file, output_path, skip_seconds=0.0):
        '''
        Convert a CSV telemetry file to a structured output format.

        Parameters:
        csv_file (str): Path to the input CSV file.
        output_path (str): Path to the output file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        self.logger.info(f"Converting csv telemetry file {csv_file} to {output_path}")
        self.telemetry_importer.read_csv(csv_file, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)
    
    def convert_zed_recorder_files(self, jsonl_file, output_path, skip_seconds=0.0):
        '''
        Convert ZED recorder files to a structured output format.

        Parameters:
        jsonl_file (str): Path to the input JSON Lines file.
        output_path (str): Path to the output file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        self.logger.info(f"Converting zed recorder files {jsonl_file} to {output_path}")
        self.telemetry_importer.read_zed_jsonl(jsonl_file, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)

    def convert_pygpmf_telemetry(self, input_json, output_path, skip_seconds=0.0):
        '''
        Convert a Pygpmf telemetry JSON file to a structured output format.

        Parameters:
        input_json (str): Path to the input telemetry JSON file.
        output_path (str): Path to the output file.
        skip_seconds (float): Number of seconds to cut from the beginning and end of the stream.
        '''
        self.logger.info(f"Converting pygpmf telemetry file {input_json} to {output_path}")
        self.telemetry_importer.read_pygpmf_json(input_json, skip_seconds=skip_seconds)
        self._dump_final_json(output_path)
