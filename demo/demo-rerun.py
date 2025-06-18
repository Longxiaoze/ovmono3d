# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis
from pycocotools.coco import COCO
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import rerun as rr
import rerun.blueprint as rrb
from pyquaternion import Quaternion
import re


def draw_annotations(boxes, labels, name="detections"):
    """
    boxes: list of length-N, each element is an (8,3) array-like of the 8 corner vertices
    labels: list of length-N of category strings
    """
    # Compute box centers as the mean of the 8 corners
    centers = [np.mean(np.array(verts).reshape(-1, 3), axis=0) for verts in boxes]
    # Compute box sizes (x,y,z extents) as max–min over the 8 corners
    sizes = [
        tuple(np.max(arr := np.array(verts).reshape(-1, 3), axis=0) -
              np.min(arr, axis=0))
        for verts in boxes
    ]
    # Use identity quaternion (no rotation)
    quaternions = [
        rr.Quaternion(xyzw=np.array([0.0, 0.0, 0.0, 1.0]))
        for _ in boxes
    ]
    # Colors look-up per category (must be defined elsewhere)
    # colors = [classname_to_color[cat] for cat in labels]

    rr.log(name, rr.Boxes3D(
        centers=centers,
        sizes=sizes,
        labels=list(labels),
        quaternions=quaternions
    ))


def do_test(args, cfg, model):

    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')
    list_of_ims = [ im for im in list_of_ims if not im.endswith('.json')]
    list_of_cats_per_img = util.load_json(args.labels_file)

    model.eval()
    
    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)



    
    for path in tqdm(list_of_ims):
        im_name = util.file_parts(path)[1]
        im = util.imread(path)
        cats = ["car", "pedestrian", "cyclist", "bus", "truck", "trailer", "motorcycle", "traffic_cone", "barrier"]
        if cats == []:
            continue
        if im is None:
            continue
        
        image_shape = im.shape[:2]  # h, w

        h, w = image_shape
        
        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2

        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point

        K = np.array([
            [focal_length, 0.0, px], 
            [0.0, focal_length, py], 
            [0.0, 0.0, 1.0]
        ])
        K = np.array([
            [1700, 0.0, 960], 
            [0.0, 1700, 620], 
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K, 'category_list': cats
        }]
        dets = model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        bbox_list = []
        catergories_list = []
        scores_list = []
        

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):



                # skip
                if score < thres:
                    continue
                bbox_list.append(corners3D.cpu().numpy().tolist())
                print("corners3D:", corners3D.cpu().numpy().tolist())
                catergories_list.append(cats[cat_idx])
                scores_list.append(score.cpu().numpy())

                cat = cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        #         print("bbox3D:", bbox3D)
        #         print('pose.tolist():', pose.tolist())

        # print("bbox_list:", bbox_list)
        # print("catergories_list:", catergories_list)
        # print("scores_list:", scores_list)
        basename = os.path.basename(path)
        m = re.match(r'.*__(\d+)\.jpg$', basename)
        if m:
            ts_micro = int(m.group(1))
            ts_sec = ts_micro * 1e-6
        else:
            ts_sec = 0.0
        rr.set_time(timeline="timestamp", timestamp=ts_sec)
        draw_annotations(bbox_list, catergories_list, name="/pcd/detections")
                
        
        print('File: {} with {} dets'.format(im_name, len(meshes)))

        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
            # rr.log(f"/camera/image_boxes", rr.Image(im_drawn_rgb.astype(np.uint8)))
            # rr.log(f"/camera/render_boxes", rr.Image(im_topdown.astype(np.uint8)))
            image_boxes = im_drawn_rgb.copy()
            render_boxes = im_topdown.copy()
            image_boxes = image_boxes[..., ::-1]  # Convert BGR to RGB

            rr.log(f"/camera/image_boxes", rr.Image(image_boxes))
            rr.log(f"/camera/render_boxes", rr.Image(render_boxes))
            if args.display:
                vis.imshow(im_concat)

            util.imwrite(im_concat, os.path.join(output_dir, im_name+'_combine.jpg'))
            # util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
            # util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
        else:
            util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    # 1. 定义 Blueprint：左右并排，左侧 3D，右侧 2D（图像）
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            # 左侧：3D 视图，显示 /pcd 路径下的点云/3D 实体
            rrb.Spatial3DView(
                name="3D",
                origin="/pcd",  # 对应 rr.log 中的实体路径
            )),
            # 右侧：2D 视图，显示 /camera 路径下的图像
        rrb.Vertical(
                rrb.Spatial2DView(origin="/camera/image_boxes",name="image+boxes"),  # 对应 rr.log 中的图像路径
                rrb.Spatial2DView(origin="/camera/render_boxes",name="render boxes"),  # 对应 rr.log 中的渲染图像
            ),
        
        # 折叠侧边栏以扩大主视图
        collapse_panels=False,
    )

    # 2. 初始化 Rerun 会话并激活 Blueprint
    rr.init(
        "ovmono3d demo",
        spawn=True,
        default_blueprint=my_blueprint
    )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument('--labels-file',  type=str, help='path to labels file', required=True)
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.2, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)
    
    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
