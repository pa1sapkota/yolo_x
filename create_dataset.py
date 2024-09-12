# Script to create the yolo-x coco format from the doclaynet dataset import os 
import json 
import shutil


def read_json(file_path:str) -> dict: 
    """
    Reads the json direcotry and returns the json data 
    """
    with open(file_path, 'r') as fp: 
        json_data = json.load(fp)
    return json_data


class DatasetCreator: 
    """
    Create YoloX format of the dataset from the Microsoft COCO  Dataset format(Doclaynet)
    """
    def __init__(self, file_path, dataset_name): 
        self.file_path = file_path 
        self.dataset_name = dataset_name
        
    
    def create_dataset_dir(self): 
        """
        Creates the Initial Dataset Directory as suggested by the yolox dataset format on the repo 
        
        
        Dataset should be 
        annotations/
        |__instances_train2017.json
        |__instances_val2017.json 
        
        train2017/
        |__img1.png 
        |__img2.png
        
        val2017/ 
        |__img1.png
        |__img2.png
        
        """
        os.makedirs(self.dataset_name, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_name, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_name,"train2017"),exist_ok=True)
        os.makedirs(os.path.join(self.dataset_name,"val2017"),exist_ok=True)
        
    def make_dataset(self): 
        self.create_dataset_dir() # Creates the dataset directory 
        
        train_json = read_json(os.path.join(self.file_path,"train.json"))
        val_json = read_json(os.path.join(self.file_path,"val.json"))
        
        for images in train_json['images']: 
            shutil.copy(f"{self.file_path}/PNG/{images['file_name']}", f"{self.dataset_name}/train2017/")
        for images in val_json['images']: 
            shutil.copy(f"{self.file_path}/PNG/{images['file_name']}", f"{self.dataset_name}/val2017/")
        
        # create a appropriate json at the annotations dir 
        with open(f"{self.dataset_name}/annotations/instances_train2017.json", 'w') as fp: 
            json.dump(train_json, fp, indent=4)
            
        with open(f"{self.dataset_name}/annotations/instances_val2017.json", 'w') as fp: 
            json.dump(val_json, fp, indent=4)
        
        
        
if __name__ == "__main__": 
    doclaynet_path = "/mnt/c/Users/FM-PC-LT-356/Documents/Testing and Learning/Small-Doclaynet/doclaynet" 
    yolox_data = DatasetCreator(doclaynet_path, "yolox_doclaynet") 
    yolox_data.make_dataset()
    print("Dataset successfully created for the Given format")