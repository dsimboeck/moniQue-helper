import numpy as np
import pygfx as gfx
import open3d as o3d
from monique_helper.transforms import alzeka2rot
from PIL import Image
from pyproj import Transformer

def plane_from_camera(cam, img, dist_plane=100, min_xyz = None):
    cmat = np.array([[1, 0, -cam["img_x0"]], 
                    [0, 1, -cam["img_y0"]],
                    [0, 0, -cam["f"]]])
    
    rmat = alzeka2rot([cam["alpha"], cam["zeta"], cam["kappa"]])
    prc_local = np.array([cam["obj_x0"], cam["obj_y0"], cam["obj_z0"]]) - min_xyz

    plane_pnts_img = np.array([[0, 0, 1],
                        [cam["img_w"], 0, 1],
                        [cam["img_w"], cam["img_h"]*(-1), 1],
                        [0, cam["img_h"]*(-1), 1]]).T
    
    plane_pnts_dir = (rmat@cmat@plane_pnts_img).T
    plane_pnts_dir = plane_pnts_dir / np.linalg.norm(plane_pnts_dir, axis=1).reshape(-1, 1)
    
    plane_pnts_obj = prc_local + dist_plane * plane_pnts_dir
    plane_faces = np.array([[3, 1, 0], [3, 2, 1]]).astype(np.uint32)
    plane_uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).astype(np.uint32)
    
    plane_geom = gfx.geometries.Geometry(indices=plane_faces, 
                                        positions=plane_pnts_obj.astype(np.float32),
                                        texcoords=plane_uv.astype(np.float32))
    
    # img_array = np.asarray(img)
    tex = gfx.Texture(img, dim=2)
    
    plane_material = gfx.MeshBasicMaterial(map=tex, side="FRONT")
    plane_mesh = gfx.Mesh(plane_geom, plane_material, visible=True)
    return plane_mesh

def img2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def nameTagGeom(lat, lon, name, tiles_epsg, min_xy, o3d_scene):

    transformer = Transformer.from_crs("EPSG:4326", tiles_epsg, always_xy=True)
    x, y = transformer.transform(lon, lat)

    local_pos_bottom = np.array([x, y, 0]) - min_xy
    local_pos_top = local_pos_bottom + np.array([0, 0, 10000])
    local_pos_terrain = raycast_terrain(local_pos_bottom, local_pos_top, o3d_scene)
    ntag_pos = local_pos_terrain + np.array([0,0,100])

    positions = np.array([local_pos_terrain, ntag_pos], dtype=np.float32)
    ntag_line = gfx.Line(gfx.Geometry(positions=positions), gfx.LineMaterial(thickness=4.0, color="#4682B4", opacity=1))

    ntag_geom = gfx.Geometry(positions=ntag_pos.astype(np.float32).reshape(1, 3))
    ntag_obj = gfx.Points(ntag_geom, gfx.PointsMaterial(color="#4682B4", size=3))        
    ntag_text = gfx.Text(geometry=None,
                          material=gfx.TextMaterial(color="#000", outline_color="#fff", outline_thickness=0.25),
                          markdown="**%s**" % (name), 
                          font_size=20, 
                          anchor="Bottom-Center", 
                          screen_space=True)
    ntag_text.local.position = ntag_pos
    ntag_obj.add(ntag_text)

    return ntag_line, ntag_obj

def raycast_terrain(ray_origin, ray_destination, o3d_scene):

    ray_direction = ray_destination - ray_origin
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    ray = o3d.core.Tensor([ray_origin.tolist() + ray_direction.tolist()], dtype=o3d.core.Dtype.Float32)
    ans = o3d_scene.cast_rays(ray)

    hit = ans['t_hit'].numpy()[0]

    if np.isinf(hit):
        return None

    hit_point = ray_origin + ray_direction * hit

    return hit_point
