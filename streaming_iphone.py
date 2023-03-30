import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import sys
import open3d as o3d
import pyfirmata as pf
from time import sleep

#from matplotlib import pyplot as plt


class DemoApp:
    def __init__(self,ard1,ard2,ard3):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.ard1 = ard1
        self.ard2 = ard2
        self.ard3 = ard3 
        
    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.
        # flag == 1
    def on_stream_stopped(self):
        print('Stream stopped')
        #cv2.destoryWindow('DEPTH')
        #cv2.destoryWindow('RGB')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
   
    def arduino_set(self):


        # To avoid overflow between python and arduino
        it1 = pf.util.Iterator(self.ard1)
        it1.start()
        it2 = pf.util.Iterator(self.ard2)
        it2.start()
        it3 = pf.util.Iterator(self.ard3)
        it3.start()

        # brake pin set with mega1 and mega2
        brak1 = self.ard1.get_pin('d:22:o')
        brak1.write(0)
        brak2 = self.ard2.get_pin('d:22:o')
        brak2.write(0)
        

        for i in range(23, 36):
            self.ard1.digital[i].mode = pf.OUTPUT
            self.ard1.digital[i].write(0)
            self.ard2.digital[i].mode = pf.OUTPUT
            self.ard2.digital[i].write(0)
            

        for i in range(2, 14):
            self.ard1.digital[i].mode = pf.PWM
            self.ard1.digital[i].write(0)    #0~11
            self.ard2.digital[i].mode = pf.PWM
            self.ard2.digital[i].write(0)   #12~23

        self.ard3.digital[2].mode = pf.PWM
        self.ard3.digital[2].write(0)

        
    
    def start_processing_stream(self):
        
        for j in range(300):
            
            for i in range(10):
               
                
                rand_pwm = np.random.random(size = (1,25))
                rand_dir = np.random.randint(0, 2, size = (1,25))
                dp = np.concatenate((rand_dir,rand_pwm), axis = 0)
                
                for k1 in range(23, 35):

                    self.ard1.digital[k1].write(rand_dir[0,k1-23]) #0 ~ 11
                    if k1 == 33 or k1 == 34:
                        self.ard2.digital[k1+1].write(rand_dir[0,k1-10])
                    else:
                        self.ard2.digital[k1].write(rand_dir[0, k1-10])
                    sleep(0.1)

                self.ard1.digital[35].write(rand_dir[0, 12])

                for k2 in range(2, 14):

                    self.ard1.digital[k2].write(rand_pwm[0,k2-2])    #0~11 > 12ㄱ개

                    
                    self.ard2.digital[k2].write(rand_pwm[0,k2+11])   #13~24 > 12개
                    
                    
                self.ard3.digital[2].write(rand_pwm[0,12]) #12
                
                self.event.wait()# Wait for new frame to arrive
                # If flag == 1 , return
                # If flag == 0 . waiting to flag == 1
                sleep(5)
                
                # Copy the newly arrived RGBD frame
                depth = self.session.get_depth_frame()
                depth = cv2.flip(depth, 1)
                rgb = self.session.get_rgb_frame()
                intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
                camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])
            
                cx = intrinsic_mat[0][2]
                cy = intrinsic_mat[1][2]
                fx = intrinsic_mat[0][0]
                fy = intrinsic_mat[1][1]
            
                pcd = []
                h, w = depth.shape
                l = h * w
                jj = np.tile(range(w), h)
                ii = np.repeat(range(h), w)
                z = depth.reshape(l)
                pcd = np.dstack([(ii - cx) * z / fx,
                             (jj - cy) * z / fy,
                             z]).reshape((l, 3))
                pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
                o3d.visualization.draw_geometries([pcd_o3d])
                sleep(1)
                
                import pyautogui
                pyautogui.press('esc')
                #template = '/Users/kweonhyuckjin/Documents/pcd_collection/pcd{}{}.csv'
                template = '/Users/kweonhyuckjin/opt/anaconda3/envs/dySurf/pcd_collection/pcd{}{}.csv'
                np.savetxt(template.format(j, i), pcd, delimiter=',')
                #template = '/Users/kweonhyuckjin/Documents/dp_collection/pwm{}{}.csv'
                template = '/Users/kweonhyuckjin/opt/anaconda3/envs/dySurf/dp_collection/pwm{}{}.csv'
                np.savetxt(template.format(j, i), dp , delimiter=',')
                
                for i in range(23, 36):

                    self.ard1.digital[i].write(0) #0 ~ 11
                    
                    if i == 33 or i ==34:
                        self.ard2.digital[i+1].write(0)
                    else:
                        self.ard2.digital[i].write(0) #13~25
                    
                    sleep(0.1)


                for i in range(2, 14):

                    self.ard1.digital[i].write(0)    #0~11 > 12ㄱ개

                    self.ard2.digital[i].write(0)   #12~23 > 12개
                    
                self.ard3.digital[2].write(0)
                sleep(10)
                i += 1
                # Postprocess it
                # if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                    
                #     rgb = cv2.flip(rgb, 1)
                    
                # depth = cv2.flip(depth, 1)
                # rgb = cv2.flip(rgb, 1)                 
                # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Show the RGBD Stream
                #cv2.imshow('RGB', rgb)
                #cv2.imshow('Depth', depth)
      
                k = cv2.waitKey(1)
                if k == 27:
                    break
                elif k == -1:
                    continue

                self.event.clear() # flag == 0
                
            j += 1
            if j%10 ==0 :
                sleep(300)
            
if __name__ == '__main__':
    print('straming2')
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
    app.on_stream_stopped()

