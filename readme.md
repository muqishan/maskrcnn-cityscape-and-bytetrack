The project is based on the merger of maskcnn and bytetrack, and their selfless open source allowed me to learn more in-depth knowledge.
1. pip install -r requirements.txt
You can use Google to search for related installation errors.
2. You need to download the cityscape dataset, pick two cities you like, keep the json file and image, and then use a small script to rename them, a001.json and a001.png
3. Then use city2labelme to convert the recording format, use the labelme2voc tool provided by labelme to convert to voc style, and move JPEGImages and SegmentationObjectPNG to the datasets path.
4. Choose your hyperparameters in train.py and run
5. Take a picture at random, then give the path in the inference, and run

The project is currently under maintenance. Since bytetrack is a bit difficult to use, you can replace the backbone network of maskrcnn in the model, where I used the pre-trained model provided by coco, and you can also customize the model file.
Update various network structures and tracking algorithms in a short period of time.
The project is maintained for a long time


From a lowly algorithm research dog
