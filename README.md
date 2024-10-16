# AppNets
An Efficient Multi-Task Fusion Network for Panoptic Driving Perception 

## The Illustration of AppNets
![image](https://github.com/user-attachments/assets/5e65acd1-bd23-4af4-b248-1cc98ba3714e)


## Contributions
•	AppNets: An end-to-end perception network achieving real-time performance on the BDD100K and SDExpressway datasets.
•	Enhanced Dataset: Augmentation of the SDExpressway dataset.
•	Effective Training Loss Function and Strategy: Balanced and optimized multitask network training.

## download
•	SDExpressway

## Results
### Traffic object detection results on SDExpressway and BDD100k datasets.
#### BDD100K
|Model          | Recall(%) | mAP50(%) | Speed(fps) |
| ------------- | --------- | -------- | ---------- |
| `YOLOP`       | 89.9      |76.3     | 223        |
| `MultiNet`      | 81.3      | 60.2     | 51     |
| `DLT-Net`     | 89.4     | 68.4       |56|  
| `HybridNets`      |91.2      | 79.0    | 220      |
| `AppNet(ours)`  | 91.7     | 79.5     | 200        
#### SDExpressway
|Model          | Recall(%) | mAP50(%) | Speed(fps) |
| ------------- | --------- | -------- | ---------- |
| `YOLOP`       | 92.1      |77.7     | 231        |
| `HybridNets`      |93.9      | 78.7    | 200      |
| `AppNet(ours)`  | 95.5     | 85.1     | 181        |
### Performance comparison on lane line segment task.
#### SDExpressway
|Model          | Acc(%) | Iou(%) | 
| ------------- | --------- | -------- | 
| `YOLOP`       | 90.4      |74.0     |
| `HybirdNets`       | 90.8     |74.3     |
| `AppNet(ours)`  | 90.6    | 75.1     |
### Performance comparison on drivable area segment task.
|Model          | Acc(%) | mIou(%) | 
| ------------- | --------- | -------- | 
| `YOLOP`       | 98.9      |97.0     |
| `HybirdNets`       | 98.7     |97.2|
| `AppNet(ours)`  | 99.2    | 98.7    |
### Comparison Experiments of Different Attention Modules
![8fdd4ee19b6df6a2c48d3e112b96951](https://github.com/user-attachments/assets/977ee82f-4a9f-4a06-a78c-4dbbddee5627)
### Comparison Experiments of Different SPPF Modules
![c807078e066f7228863fe3fd7a55b5c](https://github.com/user-attachments/assets/bee18f7f-fe53-4df4-83c0-bf89e797d65c)



## Visulization
### Traffic Object Detection Result in SDExpressway+ and BDD100k
![image](https://github.com/user-attachments/assets/af7a959f-aab9-46e3-9470-07ba07222c7f),![image](https://github.com/user-attachments/assets/0b074917-3aaa-4c32-a842-a01cc3a21845),![image](https://github.com/user-attachments/assets/85faeae3-9927-48a5-9266-cb2d5957aa8f)
![image](https://github.com/user-attachments/assets/1d35089c-5252-4786-819c-3966e90f4de7),![image](https://github.com/user-attachments/assets/817fb59f-d91d-42e3-86e1-b336677163ee),![image](https://github.com/user-attachments/assets/49130b86-259c-40c2-bdaf-39b9349b441a)

###  multitask visualization detection results
![image](https://github.com/user-attachments/assets/dc27f2e5-8942-48e3-852e-706636701c3e),![image](https://github.com/user-attachments/assets/fc99878c-a41e-41c4-aaf1-a230eaef8148),![image](https://github.com/user-attachments/assets/a717429f-fd0e-4174-819f-ea3a511cc8a9)







