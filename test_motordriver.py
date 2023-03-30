#%% inital setting
import pyfirmata as pf
from time import sleep
import numpy as np


sleep(1)

port1 = '/dev/cu.usbmodem141301' #mega1 usb port name(1:12)
port2 = '/dev/cu.usbmodem1414401' #mega2 usb port name(13:24)
port3 = '/dev/cu.usbmodem141201'

ard1 = pf.ArduinoMega(port1)
ard2 = pf.ArduinoMega(port2)
ard3 = pf.ArduinoMega(port3)

# To avoid overflow between python and arduino
it1 = pf.util.Iterator(ard1)
it1.start()
it2 = pf.util.Iterator(ard2)
it2.start()
it3 = pf.util.Iterator(ard3)
it3.start()

#streaming_iphone()

# brake pin set with mega1 and mega2
brak1 = ard1.get_pin('d:22:o')
brak1.write(0)
brak2 = ard2.get_pin('d:22:o')
brak2.write(0)

#rand_dir = np.random.randint(0, 2, size = (1,25))

#direction pins random set
'''
for i in range(23, 36):

    ard1.digital[i].mode = pf.OUTPUT
    ard1.digital[i].write(rand_dir[0,i-23])
    ard2.digital[i].mode = pf.OUTPUT
    ard2.digital[i].write(rand_dir[0,i-10])
    sleep(0.1)
'''
# for test all direction set are 0

for i in range(23, 36):

    ard1.digital[i].mode = pf.OUTPUT
    ard1.digital[i].write(0)
    ard2.digital[i].mode = pf.OUTPUT
    ard2.digital[i].write(0)

#rand_pwm = np.random.random(size = (1,25))

#pwm pins set and the value set range from 0 to 1
'''
for i in range(2, 14):
    ard1.digital[i].mode = pf.PWM
    ard1.digital[i].write(rand_pwm[0,i-2])    #0~11
    ard2.digital[i].mode = pf.PWM
    ard2.digital[i].write(rand_pwm[0,i+10])   #12~23
'''
#For test, all pwm set are 1

for i in range(2, 14):
    ard1.digital[i].mode = pf.PWM
    ard1.digital[i].write(0)    #0~11
    ard2.digital[i].mode = pf.PWM
    ard2.digital[i].write(0)   #12~23


ard3.digital[2].mode = pf.PWM
ard3.digital[2].write(0)

#%% clear

for i in range(23, 36):

    ard1.digital[i].mode = pf.OUTPUT
    ard1.digital[i].write(0)
    ard2.digital[i].mode = pf.OUTPUT
    ard2.digital[i].write(0)

for i in range(2, 14):
    ard1.digital[i].mode = pf.PWM
    ard1.digital[i].write(0)    #0~11
    ard2.digital[i].mode = pf.PWM
    ard2.digital[i].write(0)   #12~23

ard3.digital[2].mode = pf.PWM
ard3.digital[2].write(0)
#%% X_shape
x1 = [2,6, 8, 10]
x1_e = [3,4,5,7,9,11,12,13]
x1_ed = [24,25,26,28,30,32,33,34]
x2 = [5, 7, 9,13]
x2_e = [2,3,4,6,8,10,11,12]
x2_ed = [23,24,25,27,29,31,32,33]
x3 = [2]

for i in x1:
    ard1.digital[i].write(1)   
    
for i in x1_e:
    ard1.digital[i+21].write(1)
    ard1.digital[i].write(1)   

for i in x2:
    ard2.digital[i].write(1)   
    
for i in x2_e:
    ard2.digital[i+21].write(1)
    ard2.digital[i].write(1)   

ard1.digital[35].write(0)
ard3.digital[2].write(1)

#%% H_shape

x1 = [2,3,4,5,6,9]
x1_e = [7,8,10,11,12,13]
x2 = [6,9,10,11,12,13]
x2_e = [2,3,4,5,7,8]

for i in x1:
    ard1.digital[i].write(1)   
    
for i in x1_e:
    ard1.digital[i+21].write(1)
    ard1.digital[i].write(1)   

for i in x2:
    ard2.digital[i].write(1)   
    
for i in x2_e:
    ard2.digital[i+21].write(1)
    ard2.digital[i].write(1)   

ard1.digital[35].write(0)
ard3.digital[2].write(1)

#%% rectangular shape

x1 = [8,9,10,13]
x1_e = [2,3,4,5,6,7,11,12]
x2 = [2,5,6,7]

x2_e = [3,4,8,9,10,11,12,13]

for i in x1:
    ard1.digital[i+21].write(1)
    ard1.digital[i].write(1)   
    
for i in x1_e:
    ard1.digital[i+21].write(0)
    ard1.digital[i].write(1)   

for i in x2:
    ard2.digital[i+21].write(1)
    ard2.digital[i].write(1)   
    
for i in x2_e:
    ard2.digital[i+21].write(0)
    ard2.digital[i].write(1)   


ard1.digital[35].write(1)
ard3.digital[2].write(1)

#%% sangman shape
x1 = [2,4,6,8,10,12]
x1_e = [3,5,7,9,11,13]
x2 = [2,4,6,8,10,12]
x2_e = [3,5,7,9,11,13]

for i in x1:
    ard1.digital[i].write(1)   
    
for i in x1_e:
    ard1.digital[i+21].write(1)
    ard1.digital[i].write(1)   

for i in x2_e:
    ard2.digital[i].write(1)   
    
for i in x2:
    ard2.digital[i+21].write(1)
    ard2.digital[i].write(1)   

#%% + shape

x1 = [4,9,12,13]
x1_e = [2,3,5,6,7,8,10,11]
x2 = [2,3, 6,11]
x2_e = [4,5,7,8,9,10,12,13]

for i in x1:
    ard1.digital[i].write(1)   
    
for i in x1_e:
    ard1.digital[i+21].write(1)
    ard1.digital[i].write(1)   

for i in x2:
    ard2.digital[i].write(1)   
    
for i in x2_e:
    ard2.digital[i+21].write(1)
    ard2.digital[i].write(1)   
ard3.digital[2].write(1)

#%% streaming

import cv2
import streaming_iphone as si
import numpy as np
import open3d as o3d
import pyfirmata as pf
from time import sleep
import numpy as np

print('straming')

sleep(1)

port1 = '/dev/cu.usbmodem141301' #mega1 usb port name(1:12)
port2 = '/dev/cu.usbmodem1414401' #mega2 usb port name(13:24)
port3 = '/dev/cu.usbmodem141201'

ard1 = pf.ArduinoMega(port1)
ard2 = pf.ArduinoMega(port2)
ard3 = pf.ArduinoMega(port3)

app = si.DemoApp(ard1, ard2, ard3)
app.arduino_set()
app.connect_to_device(dev_idx=0)
app.start_processing_stream()
app.on_stream_stopped()

#np.savetxt('depth(H).csv', depth, delimiter=',')

 



#ard3.get_pin('d:2:p').write(rand_pwm[0,24])   #24
#%%randome

rand_pwm = np.random.random(size = (1,25))
rand_dir = np.random.randint(0, 2, size = (1,25))

for i in range(23, 35):

    ard1.digital[i].write(rand_dir[0,i-23]) #0 ~ 11
    
    ard2.digital[i].write(rand_dir[0,i-10]) #13~25
    
    sleep(0.1)


ard3.digital[35].write(rand_dir[0, 12])

for i in range(2, 14):

    ard1.digital[i].write(rand_pwm[0,i-2])    #0~11 > 12ㄱ개

    ard2.digital[i].write(rand_pwm[0,i+11])   #12~23 > 12개
    
ard3.digital[2].write(rand_pwm[0,12])

















#%%Test

for i in range(2,14):
    ard1.digital[i].write(1)
    sleep(3)
    ard1.digital[i+21].write(1)
    sleep(3)
    ard1.digital[i].write(0)
    ard1.digital[i+21].write(0)
    
for i in range(2,14):
    ard2.digital[i].write(1)
    sleep(3)
    if i == 12 or i == 13:
        ard2.digital[i+22].write(1)
    else:
        ard2.digital[i+21].write(1)
    sleep(3)
    
    ard2.digital[i].write(0)
    if i == 12 or i == 13:
        ard2.digital[i+22].write(0)
    else:
        ard2.digital[i+21].write(0)

ard3.digital[2].write(1)
sleep(3)
ard1.digital[35].write(1)
sleep(3)
ard3.digital[2].write(0)
ard1.digital[35].write(0)

#%%
for i in range(10):
   
    
    rand_pwm = np.random.random(size = (1,25))
    rand_dir = np.random.randint(0, 2, size = (1,25))
    dp = np.concatenate((rand_dir,rand_pwm), axis = 0)
    
    for k1 in range(23, 35):

        ard1.digital[k1].write(rand_dir[0,k1-23]) #0 ~ 11
        if k1 == 33 or k1 == 34:
            ard2.digital[k1+1].write(rand_dir[0,k1-10]) #23,24
        else:
            ard2.digital[k1].write(rand_dir[0, k1-10]) # 13~22
        sleep(0.1)

    ard1.digital[35].write(rand_dir[0, 12])

    for k2 in range(2, 14):

        ard1.digital[k2].write(rand_pwm[0,k2-2])    #0~11 > 12ㄱ개

        
        ard2.digital[k2].write(rand_pwm[0,k2+11])   #13~24 > 12개
        

    ard3.digital[2].write(rand_pwm[0,12]) #12
    
    print(dp)
    
    sleep(500)
    
    for i in range(23, 36):

        ard1.digital[i].write(0) #0 ~ 11
        
        if i == 33 or i ==34:
            ard2.digital[i+1].write(0)
        else:
            ard2.digital[i].write(0) #13~25
        
        sleep(0.1)


    for i in range(2, 14):

        ard1.digital[i].write(0)    #0~11 > 12ㄱ개

        ard2.digital[i].write(0)   #12~23 > 12개
        
    ard3.digital[2].write(0)
    
    i += 1

    
    sleep(10)





#%%


