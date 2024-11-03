
import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
import torch


parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()


#Create list objects to store all variables
num_of_objects = 2
lst_to_origin = []
lst_bbox = []
lst_est = []
lst_mask = []

# def run_inference(i, num, lst_to_origin_ref, lst_bbox_ref, lst_est_ref, lst_mask_ref, cam_K_ref, color_ref, depth_ref, args_ref): #Inputs are all the variables in this function

#     if i==0:
#         if len(lst_mask[num].shape)==3:
#             for c in range(3):
#                 if lst_mask[num][...,c].sum()>0:
#                     lst_mask[num] = lst_mask[num][...,c]
#                     break
#         lst_mask[num] = cv2.resize(lst_mask[num], (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)

#         pose = lst_est[num].register(K=cam_K, rgb=color, depth=depth, ob_mask=lst_mask[num], iteration=args.est_refine_iter)
#     else:
#         pose = lst_est[num].track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
#     center_pose = pose@np.linalg.inv(lst_to_origin[num])
#     return center_pose


# Initialize cuda streams
streams = [torch.cuda.Stream() for _ in range(num_of_objects)]



for i in range(num_of_objects):
    mesh_path = filedialog.askopenfilename()
    if not mesh_path:
        print("No mesh file selected")
        exit(0)
    # mesh_path = '/home/jamesl3483/Desktop/Research/live-pose/FoundationPose/demo_data/YCB_Video_Models/models/010_potted_meat_can/textured_simple.obj'

    mask_file_path = create_mask(f'mask_{i}')
    mesh = trimesh.load(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx)
    mask = cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED)

    #Append all values for future inference
    lst_to_origin.append(to_origin)
    lst_bbox.append(bbox)
    lst_est.append(est)
    lst_mask.append(mask)




#Start Camera stream
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)


cam_K = np.array([[615.37701416, 0., 313.68743896],
                   [0., 615.37701416, 259.01800537],
                   [0., 0., 1.]])
Estimating = True
time.sleep(3) 
i = 0
# Streaming loop
try:

    while Estimating:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
        if cv2.waitKey(1) == 13:
            Estimating = False
            break        
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0

        center_poses = []

        for num in range(num_of_objects):
            with torch.cuda.stream(streams[num]):
                if i==0:
                    if len(lst_mask[num].shape)==3:
                        for c in range(3):
                            if lst_mask[num][...,c].sum()>0:
                                lst_mask[num] = lst_mask[num][...,c]
                                break
                    lst_mask[num] = cv2.resize(lst_mask[num], (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)

                    pose = lst_est[num].register(K=cam_K, rgb=color, depth=depth, ob_mask=lst_mask[num], iteration=args.est_refine_iter)
                else:
                    pose = lst_est[num].track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
                center_pose = pose@np.linalg.inv(lst_to_origin[num])
                center_poses.append(center_pose)

        torch.cuda.synchronize()
        
        for num in range(num_of_objects):
            vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_poses[num], bbox=lst_bbox[num])
            vis = draw_xyz_axis(color, ob_in_cam=center_poses[num], scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)        
        i += 1
        
finally:
    pipeline.stop()