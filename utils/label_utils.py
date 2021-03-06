import configparser


def load_classes_from_csv_reader(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id

    labels = {}
    for key, value in result.items():
        labels[value] = key

    return labels


def load_classes_from_configfile(configfile):
    configs = configparser.ConfigParser()
    configs.read(configfile)
    labelmap = {int(i): str(j) for i, j in configs['LABELMAP'].items()}

    return labelmap
