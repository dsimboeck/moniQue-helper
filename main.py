import typer
from typing import List, Optional, Tuple, Union
from typing_extensions import Annotated
from rich.progress import track, Progress
import os
from enum import Enum
from monique_helper.terramesh import MeshGrid
from monique_helper.io import load_tile_json, load_terrain, save_tif, save_png, load_gtif
from monique_helper.transforms import alzeka2rot, R_ori2cv, alpha2azi
from monique_helper.geom import plane_from_camera, img2square, nameTagGeom
from osgeo import gdal, osr, ogr
import json
import string
import numpy as np
import pygfx as gfx
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
import open3d as o3d
from pyproj import Transformer
from PIL import Image
import base64
from io import BytesIO
import pandas as pd
import imageio.v3 as iio
import imageio
import math

class MeshSimplification(str, Enum):
    delatin = "delatin"

app = typer.Typer()

@app.command()
def create_mesh(dtm_path:Annotated[str, typer.Argument()], 
                out_dir:Annotated[str, typer.Argument()],
                out_name:Annotated[str, typer.Argument()],
                max_error:Annotated[float, typer.Argument()],
                extent:Annotated[tuple[float, float, float, float], typer.Option(help="Clip input DTM to (minx, miny, maxx, maxy).")] = (None, None, None, None),
                method: Annotated[MeshSimplification, typer.Option(case_sensitive=False)] = MeshSimplification.delatin,
                tile_size:Annotated[int, typer.Option(help="Size of each tile in pixels.")] = 1000
                ):
    
    allowed_characters = string.ascii_letters + string.digits + "_\\/:"
        
    if not os.path.exists(dtm_path):
        raise typer.Exit("Not a valid path to the input DTM provided.")
    
    if not os.path.isfile(dtm_path):
        raise typer.Exit("Input path does appaer to be a valid file.")
    
    if os.path.exists(out_dir):
        raise typer.Exit("%s already exists." % (out_dir))
    
    spec_chars = list(set(out_dir).difference(allowed_characters))
    if len(spec_chars) > 0:
        raise typer.Exit("The output directory contains special characters: %s" % ",".join(spec_chars))
    
    spec_chars = list(set(out_name).difference(allowed_characters))
    if len(spec_chars) > 0:
        raise typer.Exit("The output name contains special characters: %s" % ",".join(spec_chars))
    
    print("Starting to create mesh tiles:")
    tile_grid = MeshGrid(path=dtm_path, tile_size=tile_size, max_error=max_error, method=method, extent=extent)
    
    print("...snapping vertices along tile boundaries.")
    tile_grid.snap_boundaries()
    
    out_dir = os.path.normpath(out_dir)
    
    print("...saving tiles to %s." % (out_dir))
    tile_grid.save_tiles(odir=out_dir, oname=out_name)
    
@app.command()
def add_ortho(op_path:Annotated[str, typer.Argument(help="Parth to the original orthophoto.")],
              json_path:Annotated[str, typer.Argument(help="Path to the *.json created by create-mesh.")],
              op_res:Annotated[float, typer.Option(help="Output resolution of the orthophoto tiles.")] = 1
              ):
    
    print("Starting to create orthophoto tiles:")
    print("...loading %s." % (op_path))
    op_data = gdal.Open(op_path)
    
    op_proj = osr.SpatialReference(wkt=op_data.GetProjection())
    op_epsg = op_proj.GetAttrValue('AUTHORITY', 1)
        
    with open(json_path, "r") as f:
        tiles_data = json.load(f)   
    tiles_epsg = tiles_data["epsg"]    
    
    op_dir = os.path.join(os.path.dirname(json_path), "op")
    if not os.path.exists(op_dir):
        os.makedirs(op_dir)    
    
    for tile in track(tiles_data["tiles"], description="Creating OP tiles..."):
        tid = tile["tid"]
        out_path = os.path.join(op_dir, "%s.jpg" % (tid))
        
        min_xyz = tile["min_xyz"]
        max_xyz = tile["max_xyz"]
        
        bbox = [min_xyz[0], min_xyz[1], max_xyz[0], max_xyz[1]]
        
        out_srs = osr.SpatialReference()
        out_srs.ImportFromEPSG(int(tiles_epsg))
        
        inp_srs = osr.SpatialReference()
        inp_srs.ImportFromEPSG(int(op_epsg))
        
        kwargs = {'format': 'JPEG', 
                'outputBounds':bbox,
                'outputBoundsSRS':out_srs,
                'srcSRS':inp_srs,
                'dstSRS':out_srs,
                'xRes':op_res, 
                'yRes':op_res,
                'resampleAlg':'bilinear'}
        ds = gdal.Warp(out_path, op_path, **kwargs)
        del ds    

@app.command()
def render_json(camera_json:Annotated[str, typer.Argument(help="Path to the *.json containing the camera parameters.")],
                tiles_json:Annotated[str, typer.Argument(help="Path to the *.json created with create-mesh.")],
                out_dir:Annotated[str, typer.Argument(help="Path to the directory where the outputs shall be stored.")],
                xyz:Annotated[bool, typer.Option(help="If additional image with the xyz-coordinates of the scene shall be created.")] = True):
           
    gfx_scene = gfx.Scene()
    bg = gfx.Background(None, gfx.BackgroundMaterial([1, 1, 1, 1]))
    gfx_scene.add(bg)
    
    print("Loading terrain...")
    tiles_data = load_tile_json(tiles_json)
    gfx_terrain, o3d_scene = load_terrain(tiles_data)
    gfx_scene.add(gfx_terrain)
      
    with open(camera_json, "r") as json_file:
        cam_data = json.load(json_file)   
    
    for name, data in cam_data.items():
        
        print("...rendering %s." % (name))
        
        cam_w = data["img_w"]
        cam_h = data["img_h"]
        
        offscreen_canvas = OffscreenCanvas(size=(cam_w, cam_h), pixel_ratio=1)
        offscreen_renderer = gfx.WgpuRenderer(offscreen_canvas, pixel_ratio=1)            
        
        euler = np.array([data["alpha"], data["zeta"], data["kappa"]])
        rmat = alzeka2rot(euler)
        rmat_gfx = np.zeros((4,4))
        rmat_gfx[3, 3] = 1
        rmat_gfx[:3, :3] = rmat
        
        # we adjust the fov of the camera the the larger side of the image matches the fov
        if cam_h > cam_w:
            cam_fov = data["fov"]
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(cam_fov), 
                                               depth_range=(1, 100000))
        else:
            cam_fov = data["fov"] * (cam_h / cam_w)
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(cam_fov), #we adjust the vfov of the camera that it matches the extent of the image
                                               depth_range=(1, 100000))
        
        prc_local = np.array([data["X0"], data["Y0"], data["Z0"]]) - np.array(tiles_data["min_xyz"])
        gfx_camera.local.position = prc_local
        gfx_camera.local.rotation_matrix = rmat_gfx
            
        offscreen_canvas.request_draw(offscreen_renderer.render(gfx_scene, gfx_camera))    
        img_scene_arr = np.asarray(offscreen_canvas.draw())[:,:,:3]
        
        save_png(img_scene_arr, os.path.join(out_dir, name + ".png"))
                
        if xyz:
            
            print("...generating depth image.")
            
            # I HAVE NO IDEA WHY? Otherwise, using the same focal lenght, for images in portrait mode the depth image is completely off
            if cam_h > cam_w:
                cam_f = (cam_w/2.)/np.tan(cam_fov/2.)   #we use height as fov of pygfx equals the vertical field of view
            else:
                cam_f = (cam_h/2.)/np.tan(cam_fov/2.)
            
            cam_rot_cv = R_ori2cv(rmat)
            cam_tvec = np.matmul(cam_rot_cv*(-1), prc_local.reshape(3, 1))
            cam_rot_cv_tvec = np.vstack((np.concatenate([cam_rot_cv, cam_tvec], axis=-1), np.array([0, 0, 0, 1])))
            cam_o3d_extrinsic = o3d.core.Tensor(cam_rot_cv_tvec.astype(np.float32))

            cam_o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(cam_w, cam_h, 
                                                                  cam_f, cam_f,
                                                                  cam_w/2., cam_h/2.)
        
    
            rays = o3d_scene.create_rays_pinhole(intrinsic_matrix=cam_o3d_intrinsic.intrinsic_matrix, 
                                                 extrinsic_matrix=cam_o3d_extrinsic, 
                                                 width_px=cam_w, 
                                                 height_px=cam_h)
            rays = rays.reshape((cam_w*cam_h, 6))
            
            ans = o3d_scene.cast_rays(rays)
            ans_coord = rays[:,:3] + rays[:,3:]*ans['t_hit'].reshape((-1,1))
            ans_coord = ans_coord.numpy().reshape(-1, 3)
            ans_coord += np.array(tiles_data["min_xyz"])
            
            coord_arr = np.reshape(ans_coord, (cam_h, cam_w, 3))
            save_tif(coord_arr, os.path.join(out_dir, name + "_xyz.tif"))

@app.command()            
def render_gpkg(gpkg_path:Annotated[str, typer.Argument(help="Path to the *.gpkg containing the oriented cameras.")],
                out_dir: Annotated[str, typer.Argument(help="Path to the directory where the outputs shall be stored.")],
                padding:Annotated[float, typer.Option(help="Padding around historical image extent.")] = 1,
                cam: Annotated[Optional[List[str]], typer.Option(help="Name of the cameras to create output for.")] = None,
                w_hist:Annotated[bool, typer.Option(help="Create additional rendering with the historical image.")] = True,
                hist_dist: Annotated[float, typer.Option(help="Distance of the historical image from the camera.")] = 10,
                width: Annotated[int, typer.Option(help="Width in px of the output rendering. If None the width of the oriented image will be used.")] = None,
                height: Annotated[int, typer.Option(help="Height in px of the output rendering. If None the width of the oriented image will be used.")] = None,
                export_json: Annotated[bool, typer.Option(help="", hidden=True)] = False):
       
    if os.path.exists(gpkg_path):
        ds = ogr.Open(gpkg_path)
        gpkg_name = os.path.basename(gpkg_path).split(".")[0]
    else:
        raise typer.Exit("%s does not exists." % (gpkg_path))
    
    # If the file handle is null then exit
    if ds is None:
        raise typer.Exit("Failed to load %s." % (gpkg_path))
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
     
    # Select the dataset to retrieve from the GeoPackage and assign it to an layer instance called lyr.
    # The names of available datasets can be found in the gpkg_contents table.
    reg_lyr = ds.GetLayer("region")
    cam_lyr = ds.GetLayer("cameras")
    
    # Refresh the reader
    reg_lyr.ResetReading()
    cam_lyr.ResetReading()
    
    # for each feature in the layer, print the feature properties
    reg_feat = reg_lyr.GetNextFeature()
    reg_dict = reg_feat.items()
    
    cam_dict = {}
    for feat in cam_lyr:
        feat_dict = feat.items()
        if feat_dict["is_oriented"] == 1:
            cam_dict[feat_dict["iid"]] = feat_dict
    
    tiles_json = reg_dict["json_path"]
    
    gfx_scene = gfx.Scene()
    bg = gfx.Background(None, gfx.BackgroundMaterial([1, 1, 1, 1]))
    gfx_scene.add(bg)
    
    print("Loading terrain...")
    tiles_data = load_tile_json(tiles_json)
    gfx_terrain, _ = load_terrain(tiles_data)
    gfx_scene.add(gfx_terrain)
    
    if export_json:
        trans = Transformer.from_crs(int(tiles_data["epsg"]), 4326, always_xy=True)
        csv_spot_data = []
        csv_render_data = []
        canvas_h = 500
        canvas_w = 500
    
    for cid, data in cam_dict.items():
        
        if cam is not None:
            if cid not in cam:
                continue
             
        print("...rendering %s." % (cid))
        prc = np.array([data["obj_x0"], data["obj_y0"], data["obj_z0"]]) 
        prc_local = prc - np.array(tiles_data["min_xyz"])
        
        if export_json:
            prc_4326 = trans.transform(data["obj_x0"], data["obj_y0"])
        
        euler = np.array([data["alpha"], data["zeta"], data["kappa"]])
        rmat = alzeka2rot(euler)
        ior = np.array([data["img_x0"], data["img_y0"], data["f"]])
        
        img_path = data["path"]
        img_arr, _, _ = load_gtif(img_path)
        
        if export_json:
            bg_color = (255, 255, 255)
            img = Image.fromarray(img_arr)
            img.thumbnail((canvas_h, canvas_w))
            img_pad = img2square(img, background_color=bg_color)
            img_pad.save(os.path.join(out_dir, "%s_square.png" % (cid)))
                    
            img_pad_bits = BytesIO()
            img_pad.save(img_pad_bits, format="png")
            img_pad_str = "data:image/png;base64," + base64.b64encode(img_pad_bits.getvalue()).decode("utf-8")

            img_w, img_h = img.size
                    
            if img_w > img_h:
                diff = img_w-img_h
                bbox = [diff/2., 0, img_w-(diff/2.), img_h]
            else:
                diff = img_h-img_w
                bbox = [0, diff/2., img_w, img_h-(diff/2.)]

            thumb_img = img.resize((50, 50), box=bbox)
            thumb_bits = BytesIO()
            thumb_img.save(thumb_bits, format="png")
            thumb_str = "data:image/png;base64," + base64.b64encode(thumb_bits.getvalue()).decode("utf-8")
        
        hfov = data["hfov"]
        vfov = data["vfov"]
        
        img_h = data["img_h"]
        img_w = data["img_w"]
        
        canvas_h = img_h if width is None else width
        canvas_w = img_w if height is None else height
        
        offscreen_canvas = OffscreenCanvas(size=(canvas_w, canvas_h), pixel_ratio=1)
        offscreen_renderer = gfx.WgpuRenderer(offscreen_canvas) 
        
        rmat_gfx = np.zeros((4,4))
        rmat_gfx[3, 3] = 1
        rmat_gfx[:3, :3] = rmat
                
        if hfov > vfov:
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(hfov)+padding, depth_range=(1, 100000))
        else:
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(vfov)+padding, depth_range=(1, 100000))
            
        gfx_camera.local.position = prc_local
        gfx_camera.local.rotation_matrix = rmat_gfx
        
        offscreen_canvas.request_draw(offscreen_renderer.render(gfx_scene, gfx_camera))
        img_scene_arr = np.asarray(offscreen_canvas.draw())[:,:,:3]
        save_png(img_scene_arr, os.path.join(out_dir, cid + ".png"))
        
        if export_json:
            img_scene = Image.fromarray(img_scene_arr)
            # img_scene.save(os.path.join(out_dir, "%s_wo.png" % (cid)))
            img_scene_bits = BytesIO()
            img_scene.save(img_scene_bits, format="png")
            img_scene_str = "data:image/png;base64," + base64.b64encode(img_scene_bits.getvalue()).decode("utf-8")
        
        if w_hist and padding > 0:
            plane_mesh = plane_from_camera(data, img_arr, dist_plane=hist_dist, min_xyz=np.array(tiles_data["min_xyz"]))
            gfx_scene.add(plane_mesh)

            offscreen_canvas.request_draw(offscreen_renderer.render(gfx_scene, gfx_camera))
            img_scene_with_arr = np.asarray(offscreen_canvas.draw())[:,:,:3]
            save_png(img_scene_with_arr, os.path.join(out_dir, cid + "_hist.png"))
            gfx_scene.remove(plane_mesh)
            
            img_scene_with = Image.fromarray(img_scene_with_arr)           
            img_scene_with_bits = BytesIO()
            img_scene_with.save(img_scene_with_bits, format="png")
            img_scene_with_str = "data:image/png;base64," + base64.b64encode(img_scene_with_bits.getvalue()).decode("utf-8")
            
            if export_json:
                csv_render_data.append({"iid": "H" + cid,
                                        "render":img_scene_str, 
                                        "render_with":img_scene_with_str,
                                        })
                
                csv_spot_data.append({"iid":"H" + cid,
                                "image":img_pad_str,
                                "thumb":thumb_str,
                                "geom": "SRID=4326;POINT (%.6f %.6f)" % (prc_4326[0], prc_4326[1]),
                                "altitude": prc[2],
                                "hfov":hfov,
                                "vfov":vfov,
                                "alpha":euler[0],
                                "heading":alpha2azi(euler[0]),
                                "zeta":euler[1],
                                "kappa":euler[2],
                                "f":ior[2],                         
                                "archive":data["archiv"] if "archiv" in list(data.keys()) else None,
                                "copy": data["copy"] if "copy" in list(data.keys()) else None,
                                "von":"%s-01-01" % (data["jahr"]) if "jahr" in list(data.keys()) else "1111-01-01",
                                "bis":"%s-12-31" % (data["jahr"]) if "jahr" in list(data.keys()) else "1111-12-31"})
    
    if export_json:
        pd_spot = pd.DataFrame(csv_spot_data)
        pd_spot.to_json(os.path.join(out_dir, "%s_spot.json" % (gpkg_name)), orient="records", indent=4)
        
        pd_render = pd.DataFrame(csv_render_data)
        pd_render.to_json(os.path.join(out_dir, "%s_render.json" % (gpkg_name)), orient="records", indent=4)

@app.command()            
def animate_gpkg(gpkg_path:Annotated[str, typer.Argument(help="Path to the *.gpkg containing the oriented cameras.")],
                out_dir: Annotated[str, typer.Argument(help="Path to save the .gif file. Must include the exeension.")],
                padding:Annotated[float, typer.Option(help="Padding around historical image extent.")] = 1,
                cam: Annotated[Optional[List[str]], typer.Option(help="Name of the cameras to create output for.")] = None,
                dist_range: Annotated[Tuple[int, int, int, int], typer.Option(help="Rendering distance and step width; syntax: start stop min_step max_step")] = (1, 10000, 5, 500),
                width: Annotated[int, typer.Option(help="Width in px of the output rendering. If None the width of the oriented image will be used.")] = 1080,
                height: Annotated[int, typer.Option(help="Height in px of the output rendering. If None the width of the oriented image will be used.")] = 1080,
                name_tag: Annotated[Optional[List[str]], typer.Option(help="Add a name tag to rendering; syntax: 'lat,lon,tag_hight,name'; Can be used multiple times.")] = None):
    
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monique_helper", "myalpics_logo_black_text_trans_200px.png")
    logo_arr = np.array(Image.open(logo_path))
    
    logo_alpha = logo_arr[:, :, -1] 
    logo_alpha[logo_alpha > 0] = 255
    logo_alpha = logo_alpha / 255.
    
    logo_h, logo_w, logo_d = np.shape(logo_arr)
    
    if os.path.exists(gpkg_path):
        ds = ogr.Open(gpkg_path)
        gpkg_name = os.path.basename(gpkg_path).split(".")[0]
    else:
        raise typer.Exit("%s does not exists." % (gpkg_path))
    
    # If the file handle is null then exit
    if ds is None:
        raise typer.Exit("Failed to load %s." % (gpkg_path))
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
     
    # Select the dataset to retrieve from the GeoPackage and assign it to an layer instance called lyr.
    # The names of available datasets can be found in the gpkg_contents table.
    reg_lyr = ds.GetLayer("region")
    cam_lyr = ds.GetLayer("cameras")
    
    # Refresh the reader
    reg_lyr.ResetReading()
    cam_lyr.ResetReading()
    
    # for each feature in the layer, print the feature properties
    reg_feat = reg_lyr.GetNextFeature()
    reg_dict = reg_feat.items()
    
    cam_dict = {}
    for feat in cam_lyr:
        feat_dict = feat.items()
        if feat_dict["is_oriented"] == 1:
            cam_dict[feat_dict["iid"]] = feat_dict
    
    tiles_json = reg_dict["json_path"]
    
    gfx_scene = gfx.Scene()
    bg = gfx.Background(None, gfx.BackgroundMaterial([1, 1, 1, 1]))
    gfx_scene.add(bg)
    
    print("Loading terrain...")
    tiles_data = load_tile_json(tiles_json)
    gfx_terrain, o3d_scene = load_terrain(tiles_data)
    gfx_scene.add(gfx_terrain)
        
    for cid, data in cam_dict.items():
        
        if cam is not None:
            if cid not in cam:
                continue
        
        gif_path = os.path.join(out_dir, f"{cid}.gif")
        
        print("...rendering %s." % (cid))
        prc = np.array([data["obj_x0"], data["obj_y0"], data["obj_z0"]]) 
        prc_local = prc - np.array(tiles_data["min_xyz"])
        
        euler = np.array([data["alpha"], data["zeta"], data["kappa"]])
        rmat = alzeka2rot(euler)
        ior = np.array([data["img_x0"], data["img_y0"], data["f"]])
        
        img_path = data["path"]
        img_arr, _, _ = load_gtif(img_path)
                
        hfov = data["hfov"]
        vfov = data["vfov"]
        
        canvas_h = width
        canvas_w = height
        
        offscreen_canvas = OffscreenCanvas(size=(canvas_w, canvas_h), pixel_ratio=1)
        offscreen_renderer = gfx.WgpuRenderer(offscreen_canvas) 
        
        rmat_gfx = np.zeros((4,4))
        rmat_gfx[3, 3] = 1
        rmat_gfx[:3, :3] = rmat
                
        if hfov > vfov:
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(hfov)+padding, depth_range=(1, 100000))
        else:
            gfx_camera = gfx.PerspectiveCamera(fov=np.rad2deg(vfov)+padding, depth_range=(1, 100000))
            
        gfx_camera.local.position = prc_local
        gfx_camera.local.rotation_matrix = rmat_gfx

        tiles_epsg = tiles_data["epsg"]
        min_xyz=np.array(tiles_data["min_xyz"])
        min_xy = min_xyz.copy()
        min_xy[-1] = 0 

        if name_tag:  
            for ntag in name_tag:
                lat_str, lon_str, v_pos_str, name = ntag.split(",", 3)
                lat = float(lat_str.strip())
                lon = float(lon_str.strip())
                v_pos = float(v_pos_str.strip())
                name = name.strip()
                    
                ntag_line, ntag_text = nameTagGeom(lat, lon, v_pos, name, tiles_epsg, min_xy, o3d_scene)
                gfx_scene.add(ntag_line)
                gfx_scene.add(ntag_text)
            
        start_distance = dist_range[0]
        end_distance = dist_range[1]
        min_step = dist_range[2]
        max_step = dist_range[3]

        # def get_step_lin(distance):
        #     normalized = (distance - start_distance) / (end_distance - start_distance)
        #     return min_step + (max_step - min_step) * normalized**2
        
        # def get_step_log(distance):
        #     normalized = max((distance - start_distance) / (end_distance - start_distance), 1e-6)
        #     log_norm = math.log10(1 + 9 * normalized)
        #     step = min_step + (max_step - min_step) * log_norm
        #     return max(min_step, min(step, max_step))
        
        def get_step_exp(distance):
            normalized = (distance - start_distance) / (end_distance - start_distance)
            base = 5.0
            exp_norm = (base ** normalized - 1) / (base - 1)
            step = min_step + (max_step - min_step) * exp_norm
            return max(min_step, min(step, max_step))

        dist_range_variable = []
        current_dist = start_distance
        while current_dist <= end_distance:
            dist_range_variable.append(current_dist)
            current_dist += get_step_exp(current_dist)

        dist_range_variable += dist_range_variable[::-1][1:-1]

        frames = []

        with Progress() as progress:
            task1 = progress.add_task("...rendering frames.", total=len(dist_range_variable))

            for dx, dist in enumerate(dist_range_variable):
                plane_mesh = plane_from_camera(data, img_arr, dist_plane=dist, min_xyz=min_xyz)
                gfx_scene.add(plane_mesh)

                offscreen_canvas.request_draw(offscreen_renderer.render(gfx_scene, gfx_camera))
                img_scene_with_arr = np.asarray(offscreen_canvas.draw())[:,:,:3]

                img_scene_with_arr[-logo_h:, -logo_w:, 0] = (logo_arr[:, :, 0] * logo_alpha + img_scene_with_arr[-logo_h:, -logo_w:, 0] * (1 - logo_alpha)).astype(np.uint8)
                img_scene_with_arr[-logo_h:, -logo_w:, 1] = (logo_arr[:, :, 1] * logo_alpha + img_scene_with_arr[-logo_h:, -logo_w:, 1] * (1 - logo_alpha)).astype(np.uint8)
                img_scene_with_arr[-logo_h:, -logo_w:, 2] = (logo_arr[:, :, 2] * logo_alpha + img_scene_with_arr[-logo_h:, -logo_w:, 2] * (1 - logo_alpha)).astype(np.uint8)

                frames.append(img_scene_with_arr)
                gfx_scene.remove(plane_mesh)

                progress.update(task1, advance=1)

        print(f"...saving {gif_path}")
        #iio.imwrite(gif_path, frames, duration=50, loop=1)

        mp4_path = gif_path.replace('.gif', '.mp4')
        with imageio.get_writer(mp4_path, fps=30) as writer:
            for frame in frames:
                writer.append_data(frame)
        print("...done!")
        
    
if __name__ == "__main__":
    app()