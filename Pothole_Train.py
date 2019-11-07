from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Pothole")
trainer.setTrainConfig(object_names_array=["Pothole Severity Low","Pothole Severity High","Pothole Severity Medium"], batch_size=4, num_experiments=30, train_from_pretrained_model="pretrained-yolov3.h5") #download pre-trained model via https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/pretrained-yolov3.h5
trainer.trainModel()





