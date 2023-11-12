import yaml

with open('yolov5+lem+fem.yaml', 'r') as f:
    model_dict = yaml.safe_load(f)

for i, (f, n, m, args) in enumerate(model_dict['backbone'] + model_dict['head']):
    if m == 'areaselect':
        print('Areaselect module found in model')
        break
else:
    print('Areaselect module not found in model')
