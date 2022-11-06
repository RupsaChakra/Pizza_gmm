# Pizza_gmm
Folder structure of the repository:


            -old_ver_pizza is the folder under which the repository al-mdn which is the main folder containing all the files exsists.
            -The dataset is to be added under the data folder with the following directory structure: data->real_pizza_voc(main folder under data)->pizza_data(under real_pizza_voc), 
            under which the Annotations folder, the ImageSets folder,the JPEGImages folder, file.txt are placed.
            -create  image_list,eval,weights folder.
            -create updated_anno folder under the pizza_data folder.
            - file.txt is the file containing indices of the labeled images and in the the active learning script line 163, labeled_2.txt is the placeholder 
            for the respective file to be initially started with during first round of training.
            -Under the weights folder add the vgg16_reducedfc.pth added in the shared folder for the project.
            -The mAP values are saved under the eval folder.
            
            
 Changes made from the original repository:
 
            -The new dataloader carries the same name as the python script for the original repository but has been changed according to the pizza dataset.
            -Changes made to resume training.
            -State dictionary on resumption was causing errors which has been fixed using try except blocks.
            -Afer active learning loop list of new labeled and unlabeled dataset indices written to image_list folder.
            - After completion of the cycle, the script checks for annotations in the updated_anno folder under pizza data and if new annotations are available, resumes the code flow.
             -The mAP values are saved under the eval folder.
 
 
 Environment Installations:
            
            -visdom
            -opencv-python
            -pycocotools
            -scikit-image
            -scikit-learn
            -tensorboard
            -tqdm
 pytorch --version :1.4.0
 
 cuda : 10.1
 
 
#Training: 

            -Active learning
            CUDA_VISIBLE_DEVICES=<GPU_ID> python train_ssd_gmm_active_learining.py
 
