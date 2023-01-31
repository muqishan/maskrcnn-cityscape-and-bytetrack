import os
import json
import shutil
import base64

path = r'G:\cproject\Bubble_tracking\datasets\temp'  # Cityscape dir
ttt = r'G:\cproject\Bubble_tracking\datasets\ttt'    # 目标文件夹
json_path = r'G:\cproject\Bubble_tracking\datasets\jsons'  # 只保留json文件
json_file = list()
[json_file.append(file) if file.endswith('.json') else None for file in os.listdir(path)]
for json_ in json_file:
    result = {
        'version': '5.1.1',
        'shapes': None,
        'imagePath': None,
        'imageDate': None,
        'imageHeight': None,
        'imageWidth': None,
    }
    with open(os.path.join(path, json_), "r") as f:
        shapes = []
        row_data = json.load(f)
        for lp in row_data['objects']:
            d = {
                'label': lp['label'],
                'points': lp['polygon'],
            }
            shapes.append(d)
    img_name = json_.replace('.json', '.png')
    with open(os.path.join(path, img_name), 'rb') as ff:
        imgdata = str(base64.b64encode(ff.read()), encoding='utf-8')
    path_list = os.path.join(path, img_name).split('\\')[-2:]
    # copy img
    # shutil.copyfile(os.path.join(path, img_name), os.path.join(ttt, img_name))

    result['shapes'] = shapes
    result['imageData'] = imgdata
    result['imageWidth'] = row_data['imgWidth']
    result['imageHeight'] = row_data['imgHeight']
    result['imagePath'] = '..'+'\\'+path_list[0]+'\\'+path_list[1]
    print(json_)
    data = json.dumps(result, ensure_ascii=False,indent=4)
    with open(os.path.join(json_path, json_), 'w') as fff:
        fff.write(data)
    # print()
