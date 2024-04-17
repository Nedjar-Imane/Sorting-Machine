# Real-Time Solid Waste Sorting Machine Based on Deep Learning
We propose a machine based on deep learning and an embedded system to collect and sort trash in real-time. This machine aims to segregate plastic bottles and paper using image recognition with a convolutional neural network model implemented in Raspberry Pi. 

## Machine composition
The machine is composed of electronic component box and two parts. The upper part is the processing, which includes object detection, identification, and classification. The lower part contains the containers, one for paper and the other for plastic.   
![image](https://user-images.githubusercontent.com/25481084/216020859-f6b10155-80cf-44ac-bd7a-ed1cfa4f3a0d.png)

### Electronic component box
We have set up this box to protect the electronic components that we have used: Raspberry Pi, Pi Camera, and servomotors.  
![image](https://user-images.githubusercontent.com/25481084/216021733-c2270364-013e-4fc4-ba5e-1821a93c839f.png)
### The upper part of the machine
•	Window  
In the upper part, there is a window to throw the waste. The obstacle avoidance sensor (E18-D80NK) is fixed at the top of the window to detect objects.
The door of this window moves thanks to the servo motor (metal gears RG996R).   
•	Sorting board    
Once the sensor detects the object, the door opens, and the user throws his waste. There is the sorting board inside the box. It turns right when the waste is plastic, otherwise it turns left.  
•	Box of pens  
We have developed the machine for the educational institutions. In order to encourage students to recycle waste, when a student throws a fixed number of plastic bottles, the box of pens opens for him to take a pen. We have used servomotor to open the box . 
### The lower part of the machine
This part is designed to sort the waste into two containers, one for plastic and the other for paper. 

## Material prerequisites
•Raspberry Pi 3 Model B+      
![image](https://user-images.githubusercontent.com/25481084/216034248-9df339bf-2ba2-4847-ae50-e2d7d1d80bd6.png)  
•Servo Motor metal gears RG996R  
![image](https://user-images.githubusercontent.com/25481084/216032842-7105f8a6-a7ae-4b00-892f-51d1b92d263e.png)  
•Obstacle avoidance sensor (E18-D80NK)    
![image](https://user-images.githubusercontent.com/25481084/216033578-16db2294-d499-4535-b91b-d4149a90ca25.png)  
•Raspberry Pi Camera      
![image](https://user-images.githubusercontent.com/25481084/216034943-ce1e2ec4-3a63-4b2b-9d1c-93b4a413cb71.png)  
•I2C-LCD       
![image](https://user-images.githubusercontent.com/25481084/216035984-f78f5b47-afe1-4e7f-a71a-58d5232f2a90.png)



