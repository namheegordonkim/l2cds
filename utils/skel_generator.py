import xml.etree.ElementTree as ET

import numpy as np


def generate_skel_with_params(skel_template='./custom_dart_assets/walker2d_reg.skel', mass_scaling=1.0, volume_scaling=1.0,
                              save_path=''):
    # initialize template element
    etree = ET.parse(skel_template)
    root = etree.getroot()

    # locate walker skeleton
    walker_elem = root.find(".//skeleton[@name='walker']")

    # locate body parts (we will add more as we parametrize)
    h_pelvis_elem = walker_elem.find(".//body[@name='h_pelvis']")
    h_pelvis_aux_elem = walker_elem.find(".//body[@name='h_pelvis_aux']")
    h_pelvis_aux2_elem = walker_elem.find(".//body[@name='h_pelvis_aux2']")

    h_thigh_elem = walker_elem.find(".//body[@name='h_thigh']")
    h_shin_elem = walker_elem.find(".//body[@name='h_shin']")
    h_foot_elem = walker_elem.find(".//body[@name='h_foot']")

    h_thigh_left_elem = walker_elem.find(".//body[@name='h_thigh_left']")
    h_shin_left_elem = walker_elem.find(".//body[@name='h_shin_left']")
    h_foot_left_elem = walker_elem.find(".//body[@name='h_foot_left']")

    feet = [h_foot_elem, h_foot_left_elem]

    # apply scaling to body parts
    height_elems = walker_elem.findall(".//height")
    radius_elems = walker_elem.findall(".//radius")
    mass_elems = walker_elem.findall(".//mass")

    # volume scaling
    for h, r in zip(height_elems, radius_elems):
        original_height = float(h.text)
        original_radius = float(r.text)
        h.text = str(original_height * volume_scaling)
        r.text = str(original_radius * volume_scaling)

    # mass scaling
    for m in mass_elems:
        original_mass = float(m.text)
        m.text = str(original_mass * mass_scaling)

    # apply displacements bottom up
    # assume complete symmetry for now
    foot_height = get_height(h_foot_elem)
    shin_height = get_height(h_shin_elem)
    thigh_height = get_height(h_thigh_elem)
    pelvis_height = get_height(h_pelvis_elem)

    # print('foot_height: {:f}'.format(foot_height))
    # print('shin_height: {:f}'.format(shin_height))
    # print('thigh_height: {:f}'.format(foot_height))
    # print('pelvis_height_height: {:f}'.format(foot_height))

    foot_vertical_disp = foot_height / 2
    foot_horizontal_disp = foot_height / 2
    shin_vertical_disp = shin_height + foot_vertical_disp
    thigh_vertical_disp = thigh_height + shin_vertical_disp
    pelvis_vertical_disp = pelvis_height / 2 + thigh_vertical_disp

    # apply vertical displacement
    set_global_vertical_displacement(h_foot_elem, foot_vertical_disp)
    set_global_vertical_displacement(h_foot_left_elem, foot_vertical_disp)
    # set_local_vertical_displacement(h_foot_elem, -foot_height / 2)
    # set_local_vertical_displacement(h_foot_left_elem, -foot_height / 2)
    set_local_horizontal_displacement(h_foot_elem, foot_horizontal_disp)
    set_local_horizontal_displacement(h_foot_left_elem, foot_horizontal_disp)

    set_global_vertical_displacement(h_shin_elem, shin_vertical_disp)
    set_global_vertical_displacement(h_shin_left_elem, shin_vertical_disp)
    set_local_vertical_displacement(h_shin_elem, -shin_height / 2)
    set_local_vertical_displacement(h_shin_left_elem, -shin_height / 2)

    set_global_vertical_displacement(h_thigh_elem, thigh_vertical_disp)
    set_global_vertical_displacement(h_thigh_left_elem, thigh_vertical_disp)
    set_local_vertical_displacement(h_thigh_elem, -thigh_height / 2)
    set_local_vertical_displacement(h_thigh_left_elem, -thigh_height / 2)

    set_global_vertical_displacement(h_pelvis_elem, pelvis_vertical_disp)
    set_global_vertical_displacement(h_pelvis_aux_elem, pelvis_vertical_disp)
    set_global_vertical_displacement(h_pelvis_aux2_elem, pelvis_vertical_disp)

    etree._setroot(root)
    etree.write(save_path)

    return etree


def get_height(elem):
    return float(elem.find('.//height').text)


def set_global_vertical_displacement(elem, disp):
    trans_elem = elem.find('./transformation')
    trans_arr = np.asarray(trans_elem.text.split(), dtype=np.float)
    trans_arr[1] = disp
    trans_elem.text = " ".join([str(i) for i in trans_arr])


def set_local_vertical_displacement(elem, disp):
    trans_elem = elem.find('./visualization_shape/transformation')
    trans_arr = np.asarray(trans_elem.text.split(), dtype=np.float)
    trans_arr[1] = disp
    trans_elem.text = " ".join([str(i) for i in trans_arr])

    trans_elem = elem.find('./collision_shape/transformation')
    trans_arr = np.asarray(trans_elem.text.split(), dtype=np.float)
    trans_arr[1] = disp
    trans_elem.text = " ".join([str(i) for i in trans_arr])


def set_local_horizontal_displacement(elem, disp):
    trans_elem = elem.find('./visualization_shape/transformation')
    trans_arr = np.asarray(trans_elem.text.split(), dtype=np.float)
    trans_arr[0] = disp
    trans_elem.text = " ".join([str(i) for i in trans_arr])

    trans_elem = elem.find('./collision_shape/transformation')
    trans_arr = np.asarray(trans_elem.text.split(), dtype=np.float)
    trans_arr[0] = disp
    trans_elem.text = " ".join([str(i) for i in trans_arr])


# if __name__ == "__main__":
#     generated_skel_path = os.path.abspath('./custom_dart_assets/generated.skel')
#     volume_scaling = float(sys.argv[1])
#     mass_scaling = float(sys.argv[2])
#     print("Generate walker2d skeleton with scaling constant {:f}".format(volume_scaling))
#
#     generate_skel_with_params(skel_template='./custom_dart_assets/walker2d_reg.skel', volume_scaling=volume_scaling,
#                               mass_scaling=mass_scaling, save_path=generated_skel_path)
#
#     env = Walker2DTracking(skel_name=generated_skel_path, disable_viewer=False)
#     for i in range(1000):
#         env.step(np.ones(6))
#         env.render()
#         time.sleep(0.05)
