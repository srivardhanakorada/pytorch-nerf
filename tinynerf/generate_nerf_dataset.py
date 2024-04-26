import numpy as np
from pyrr import Matrix44
from renderer import gen_rotation_matrix_from_cam_pos, Renderer
from renderer_settings import *
SHAPENET_DIR = "/data/home1/saichandra/Vardhan/projectAIP/pytorch-nerf/ShapeNet"
def main():
    renderer = Renderer(
        camera_distance=CAMERA_DISTANCE,
        angle_of_view=ANGLE_OF_VIEW,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    img_size = 100
    focal = (img_size / 2) / np.tan(np.radians(ANGLE_OF_VIEW) / 2)
    obj = "1a2d2208f73d0531cec33e62192b66e5"
    cat = "03790512"
    obj_mtl_path = f"{SHAPENET_DIR}/{cat}/{obj}/models/model_normalized"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    init_cam_pos = np.array([0, 0, CAMERA_DISTANCE])
    target = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    samps = 800
    imgs = []
    poses = []
    for _ in range(samps):
        xyz = np.random.normal(size=3)
        xyz /= np.linalg.norm(xyz)
        R = gen_rotation_matrix_from_cam_pos(xyz)
        eye = tuple((R @ init_cam_pos).flatten())
        look_at = Matrix44.look_at(eye, target, up)
        renderer.prog["VP"].write(
            (look_at @ renderer.perspective).astype("f4").tobytes()
        )
        renderer.prog["cam_pos"].value = eye
        image = renderer.render(0.5, 0.5, 0.5).resize((img_size, img_size))
        imgs.append(np.array(image))
        pose = np.eye(4)
        pose[:3, :3] = np.array(look_at[:3, :3])
        pose[:3, 3] = -look_at[:3, :3] @ look_at[3, :3]
        poses.append(pose)
    imgs = np.stack(imgs)
    poses = np.stack(poses)
    np.savez(
        f"{obj}.npz",
        images=imgs,
        poses=poses,
        focal=focal,
        camera_distance=CAMERA_DISTANCE,
    )
    print("DONE!")
if __name__ == "__main__":
    main()