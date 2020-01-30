from PIL import Image, ImageChops
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import vol_to_surf
import os.path as osp

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def paste_images(im_list,out_file):
    widths, heights = zip(*(i.size for i in im_list))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height),(255,255,255,255))
    x_offset = 0
    for im in im_list:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    new_im.save(out_file)

def MNIvol_to_surf_pngs(vol,out_prefix,out_dir, write_combined=True, delete_single_views=False, vmax=20, threshold=0.1, alpha=0.7):
    fsaverage = fetch_surf_fsaverage(mesh='fsaverage5')
    aux_texture_right = vol_to_surf(vol,fsaverage.pial_right)
    aux_texture_left  = vol_to_surf(vol,fsaverage.pial_left)
    
    rLAT_view_path = osp.join(out_dir,out_prefix+'_Right_Lateral.png')
    lLAT_view_path = osp.join(out_dir,out_prefix+'_Left_Lateral.png')
    rMED_view_path = osp.join(out_dir,out_prefix+'_Right_Medial.png')
    lMED_view_path = osp.join(out_dir,out_prefix+'_Left_Medial.png')
    
    plot_surf_stat_map(fsaverage.infl_right, aux_texture_right, hemi='right',title='', view='lateral', colorbar=False, vmax=vmax, threshold=threshold, alpha=alpha, bg_map=fsaverage.sulc_right, darkness=0.8, output_file=rLAT_view_path)
    plot_surf_stat_map(fsaverage.infl_left,  aux_texture_left,  hemi='left', title='', view='lateral', colorbar=False, vmax=vmax, threshold=threshold, alpha=alpha, bg_map=fsaverage.sulc_left,  darkness=0.8, output_file=lLAT_view_path)
    plot_surf_stat_map(fsaverage.infl_right, aux_texture_right, hemi='right',title='', view='medial', colorbar=False,  vmax=vmax, threshold=threshold, alpha=alpha, bg_map=fsaverage.sulc_right, darkness=0.8, output_file=rMED_view_path)
    plot_surf_stat_map(fsaverage.infl_left,  aux_texture_left,  hemi='left', title='', view='medial', colorbar=False,  vmax=vmax, threshold=threshold, alpha=alpha, bg_map=fsaverage.sulc_left,  darkness=0.8, output_file=lMED_view_path)

    aux_image_rightLAT = trim(Image.open(rLAT_view_path))
    aux_image_leftLAT  = trim(Image.open(lLAT_view_path))
    aux_image_rightMED = trim(Image.open(rMED_view_path))
    aux_image_leftMED  = trim(Image.open(lMED_view_path))
    aux_image_rightLAT.save(rLAT_view_path)
    aux_image_leftLAT.save(lLAT_view_path)
    aux_image_rightMED.save(rMED_view_path)
    aux_image_leftMED.save(lMED_view_path)
    if write_combined:
        frame_path     = osp.join(out_dir,out_prefix+'_SurfView.png')
        paste_images([aux_image_leftLAT, aux_image_rightLAT, aux_image_rightMED, aux_image_leftMED],frame_path)
    if delete_single_views:
        os.remove(rLAT_view_path)
        os.remove(lLAT_view_path)
        os.remove(rMED_view_path)
        os.remove(lMED_view_path)
        os.remove(square_bot_path)
        os.remove(square_top_path)