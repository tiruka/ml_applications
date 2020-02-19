def rgb_to_lab(rgb):
    assert rgb.type == 'uint8'
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def lab_to_rgb(lab):
    assert lab.type == 'uint8'
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)