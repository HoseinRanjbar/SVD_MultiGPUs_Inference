import numpy as np

from ..utils.tools import GlobalState, DistController
from .plugins import torch, ModulePlugin, UNetPlugin, GroupNormPlugin, ConvLayerPlugin, AttentionPlugin, Conv3DPligin, dist

class DistWrapper(object):
    def __init__(self, pipe, dist_controller: DistController, config) -> None:
        super().__init__()
        self.pipe = pipe
        self.dist_controller = dist_controller
        self.config = config
        self.global_state = GlobalState({
            "dist_controller": dist_controller
        })
        self.plugin_mount()

    def switch_plugin(self, plugin_name, enable):
        if plugin_name not in self.plugins: return
        for moudule_id in self.plugins[plugin_name]:
            moudle: ModulePlugin = self.plugins[plugin_name][moudule_id]
            moudle.set_enable(enable)
    
    def config_plugin(self, plugin_name, config):
        if plugin_name not in self.plugins: return
        for moudule_id in self.plugins[plugin_name]:
            moudle: ModulePlugin = self.plugins[plugin_name][moudule_id]
            moudle.update_config(config)

    
    def plugin_mount(self):
        self.plugins = {}
        self.unet_plugin_mount()
        self.attn_plugin_mount()
        # self.group_norm_plugin_mount()
        # self.conv_3d_plugin_mount()
        # Conv3d and Conv layer can only be used one at a time
        #self.conv_plugin_mount()

    def group_norm_plugin_mount(self):
        self.plugins['group_norm'] = {}
        group_norms = []
        for module in self.pipe.unet.named_modules():
            if ('temp_' in module[0] or 'transformer_in' in module[0]) and module[1].__class__.__name__ == 'GroupNorm':
                group_norms.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(group_norms)} group norms')
        for i, group_norm in enumerate(group_norms):
            plugin_id = 'group_norm', i
            self.plugins['group_norm'][plugin_id] = GroupNormPlugin(group_norm, plugin_id, self.global_state)
            
    def conv_plugin_mount(self):
        self.plugins['conv_layer'] = {}
        convs = []
        for module in self.pipe.unet.named_modules():
            if ('temp_' in module[0] or 'transformer_in' in module[0]) and module[1].__class__.__name__ == 'TemporalConvLayer':
                convs.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(convs)} convs')
        for i, conv in enumerate(convs):
            plugin_id = 'conv_layer', i
            self.plugins['conv_layer'][plugin_id] = ConvLayerPlugin(conv, plugin_id, self.global_state)

    def conv_3d_plugin_mount(self):
        self.plugins['conv_3d'] = {}
        conv3d_s = []
        for module in self.pipe.unet.named_modules():
            if ('temp_' in module[0] or 'transformer_in' in module[0]) and module[1].__class__.__name__ == 'Conv3d':
                conv3d_s.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(conv3d_s)} conv3d_s')
        for i, conv in enumerate(conv3d_s):
            plugin_id = 'conv_3d', i
            self.plugins['conv_3d'][plugin_id] = Conv3DPligin(conv, plugin_id, self.global_state)


    def attn_plugin_mount(self):
        self.plugins['attn'] = {}
        attns = []
        for module in self.pipe.named_modules():
            if ('temp_' in module[0] or 'transformer_in' in module[0]) and module[1].__class__.__name__ == 'Attention':
                attns.append(module[1])
        if self.dist_controller.is_master:
            print(f'Found {len(attns)} attns')
        for i, attn in enumerate(attns):
            plugin_id = 'attn', i
            self.plugins['attn'][plugin_id] = AttentionPlugin(attn, plugin_id, self.global_state)

    def unet_plugin_mount(self):
        self.plugins['unet'] = UNetPlugin(
            self.pipe.unet,
            ('unet', 0),
            self.global_state
        )
    
    def inference(self,config,pipe_configs,plugin_configs,additional_info):

        self.plugin_mount()

        self.global_state.set("plugin_configs", plugin_configs)

        video_frames = self.pipe(
            pipe_configs['ref_image'],
            pipe_configs['video_clip'],  # Each GPU gets its own chunk of frames
            decode_chunk_size= pipe_configs['decode_chunk_size'],
            num_frames=len(pipe_configs['num_frames']),
            motion_bucket_id= pipe_configs['motion_bucket_id'],
            fps= pipe_configs['fps'],
            controlnext_cond_scale= pipe_configs['controlnext_cond_scale'],
            width= pipe_configs['width'],
            height= pipe_configs['height'],
            min_guidance_scale= pipe_configs['min_guidance_scale'],
            max_guidance_scale= pipe_configs['max_guidance_scale'],
            frames_per_batch= pipe_configs['frames_per_batch'],
            num_inference_steps= pipe_configs['num_inference_steps'],
            overlap= pipe_configs['overlap']
        ).frames[0]

        #video_frames = torch.tensor(video_frames, dtype=torch.float16, device=self.dist_controller.device)
        video_frames = [torch.tensor(np.array(img)).to(self.dist_controller.device) for img in video_frames]

        print(f"Rank {self.dist_controller.rank} finished inference. Result: {video_frames.shape}")

        all_frames = [torch.zeros_like(video_frames, dtype=torch.float16) for _ in range(self.dist_controller.world_size)
        ] if self.dist_controller.is_master else None

        dist.gather(video_frames, all_frames, dst=0)

        return all_frames


