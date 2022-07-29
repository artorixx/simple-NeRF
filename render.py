import bpy
import os
import numpy as np
import json

VIEWS = 100
RESOLUTION = 128
RESULTS_PATH = os.path.abspath('twobox')
mode='val'
if not os.path.exists(RESULTS_PATH):
    os.makedirs(os.path.join(RESULTS_PATH, mode),exist_ok=True)

# Render Setting
scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.image_settings.file_format = str('PNG')
scene.render.film_transparent = True
scene.render.use_persistent_data = True

# Camera Setting
cam = scene.objects['Camera']
cam.location = (0, 4.0, 0.0)

# Camera Parent
camera_parent = bpy.data.objects.new("CameraParent", None)
camera_parent.location = (0, 0, 0)
scene.collection.objects.link(camera_parent)
cam.parent = camera_parent

# Camera Constraint
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
cam_constraint.target = camera_parent


# Data to store in JSON file
out_data = {'camera_angle_x': bpy.data.objects['Camera'].data.angle_x, 'frames': []}

for i in range(VIEWS):

    # Random Camera Rotation
    camera_parent.rotation_euler = np.array([np.random.uniform(np.pi / 12, np.pi / 2), 0, np.random.uniform(0, 2 * np.pi)])

    # Rendering
    filename = 'r_{0:03d}'.format(i)
    scene.render.filepath = os.path.join(RESULTS_PATH, mode, filename)
    bpy.ops.render.render(write_still=True)

    # add frame data to JSON file
    def listify_matrix(matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list
    frame_data = {
        'file_path': './'+mode+'/' + filename,
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

with open(os.path.join(RESULTS_PATH, 'transforms_'+mode+'.json'), 'w') as out_file:
    json.dump(out_data, out_file, indent=4)