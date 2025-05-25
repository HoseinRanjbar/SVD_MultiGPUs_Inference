import time
import torch
import json
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np
import torch.distributed as dist  # Handles distributed inference
import torch.multiprocessing as mp
from PIL import Image
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from pipeline.wrapper import DistWrapper
from models.controlnext_vid_svd import ControlNeXtSDVModel
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from transformers import CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from decord import VideoReader
import argparse
from utils.pre_process import preprocess
from utils.tools import DistController


# Function to split frames into chunks for multiple GPUs
def split_video_frames(frames, num_gpus):
    chunk_size = len(frames) // num_gpus
    return [frames[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate, 
                         ffmpeg_params=["-crf", "18", "-preset", "slow"])

def save_vid_side_by_side(batch_output, validation_control_images, output_folder, fps):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    video_path = output_folder+'/test_1.mp4'
    final_images = []
    outputs = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(video_path, final_images, fps=fps)

    output_path = output_folder + "/output.mp4"
    write_mp4(output_path, outputs, fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Diffusion XL for InstructPix2Pix.")

    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None,required=True)
    parser.add_argument("--validation_control_images_folder",type=str,default=None,required=False,)
    parser.add_argument("--validation_control_video_path",type=str,default=None,required=False,)
    parser.add_argument("--output_dir",type=str,default=None,required=True)
    parser.add_argument("--height",type=int,default=768,required=False)
    parser.add_argument("--width",type=int,default=512,required=False)
    parser.add_argument("--guidance_scale",type=float,default=2.,required=False)
    parser.add_argument("--num_inference_steps",type=int,default=25,required=False)
    parser.add_argument("--controlnext_path",type=str,default=None,required=True)
    parser.add_argument("--unet_path",type=str,default=None,required=True)
    parser.add_argument("--max_frame_num",type=int,default=50,required=False)
    parser.add_argument("--ref_image_path",type=str,default=None,required=True)
    parser.add_argument("--batch_frames",type=int,default=14,required=False)
    parser.add_argument("--overlap",type=int,default=4,required=False)
    parser.add_argument("--sample_stride",type=int,default=2,required=False)
    parser.add_argument("--config", type=str, default='config.yaml')
    args = parser.parse_args()
    return args


def run_inference(rank, world_size, validation_control_images, ref_image, config, args):
    
    dist_controller = DistController(rank, world_size, config)
    # Load models
    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True).to(rank)
    controlnext = ControlNeXtSDVModel().to(rank)
    controlnext.load_state_dict(torch.load(args.controlnext_path, map_location=f"cuda:{rank}"))
    unet.load_state_dict(torch.load(args.unet_path, map_location=f"cuda:{rank}"), strict=False)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder").to(rank)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae").to(rank)

    pipeline = StableVideoDiffusionPipelineControlNeXt.from_pretrained(
        args.pretrained_model_name_or_path, controlnext=controlnext, unet=unet, vae=vae, image_encoder=image_encoder).to(rank)
    
    dist_pipe = DistWrapper(pipeline, dist_controller, config)

    dist.barrier()  # Synchronize all GPUs before inference starts
    # Process assigned chunk of frames
    pipeline_config = {
        'validation_control_video_path' : args.validation_control_video_path,
        'ref_image' : ref_image,
        'video_clip' : validation_control_images[rank],
        'decode_chunk_size' : 2,
        'num_frames' : len(validation_control_images[rank]),
        'motion_bucket_id' : 127.0,
        'fps' : 7,
        'controlnext_cond_scale' : 1.0,
        'width' : args.width,
        'height' : args.height,
        'min_guidance_scale' : args.guidance_scale,
        'max_guidance_scale' : args.guidance_scale,
        'frames_per_batch' : args.batch_frames,
        'num_inference_steps' : args.num_inference_steps,
        'overlap' : args.overlap
    }

    plugin_configs = config['plugin_configs']

    start = time.time()
    gathered_frames = dist_pipe.inference(
        config,
        pipeline_config,
        plugin_configs,
        additional_info={
            "full_config": config})
    
    fps =VideoReader(args.validation_control_video_path).get_avg_fps()  // args.sample_stride
    if rank == 0:
        # Flatten the list and convert tensors to NumPy images
        final_images = [frame.cpu().numpy() for frames in gathered_frames for frame in frames]
        
        # Convert NumPy arrays back to PIL images for video generation
        final_images_pil = [Image.fromarray(img) for img in final_images]
        
        # Save the video with correctly ordered frames
        pose_images = [pose_frame for chunk in validation_control_images for pose_frame in chunk]
        save_vid_side_by_side([final_images_pil], pose_images[:world_size*len(validation_control_images[0])], args.output_dir, fps=fps)

    dist.barrier()
    dist.destroy_process_group()
    
    

# Main script
def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    # Load video frames and reference image
    validation_control_images, ref_image = preprocess(
        args.validation_control_video_path, args.ref_image_path,
        width=args.width, height=args.height,
        max_frame_num=args.max_frame_num, sample_stride=args.sample_stride)
    
    # Split video frames into chunks per GPU
    frame_chunks = split_video_frames(validation_control_images, num_gpus)

    with open(parse_args().config, "r") as f:
        config = json.load(f)

    # Start multiprocessing for each GPU
    mp.spawn(run_inference, args=(num_gpus, frame_chunks, ref_image, config, args), nprocs=num_gpus)

if __name__ == "__main__":
    main()



