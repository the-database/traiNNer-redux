import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from ..archs import build_network
from ..losses import build_loss
from ..losses.loss_util import get_refined_artifact_map
from ..metrics import calculate_metric
from ..utils import get_root_logger, imwrite, tensor2img
from ..utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', None)
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # define network net_d
        self.net_d = None
        net_d_opt = self.opt.get('network_d', None)
        if net_d_opt is not None:
            self.net_d = build_network(net_d_opt)
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):

        self.net_g.train()
        if self.net_d is not None:
            self.net_d.train()

        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('mssim_opt'):
            self.cri_mssim = build_loss(train_opt['mssim_opt']).to(self.device)
        else:
            self.cri_mssim = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('dists_opt'):
            self.cri_dists = build_loss(train_opt['dists_opt']).to(self.device)
        else:
            self.cri_dists = None

        if train_opt.get('contextual_opt'):
            self.cri_contextual = build_loss(train_opt['contextual_opt']).to(self.device)
        else:
            self.cri_contextual = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if train_opt.get('luma_opt'):
            self.cri_luma = build_loss(train_opt['luma_opt']).to(self.device)
        else:
            self.cri_luma = None

        if train_opt.get('hsluv_opt'):
            self.cri_hsluv = build_loss(train_opt['hsluv_opt']).to(self.device)
        else:
            self.cri_hsluv = None

        if train_opt.get('avg_opt'):
            self.cri_avg = build_loss(train_opt['avg_opt']).to(self.device)
        else:
            self.cri_avg = None

        if train_opt.get('bicubic_opt'):
            self.cri_bicubic = build_loss(train_opt['bicubic_opt']).to(self.device)
        else:
            self.cri_bicubic = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        # GAN loss, network_d, optim_d must be all enabled, or all disabled
        gan_components = [self.cri_gan, self.net_d, self.opt["train"].get("optim_d", None)]
        all_enabled = all(gan_components)
        all_disabled = all(component is None for component in gan_components)

        if not (all_enabled or all_disabled):
            raise ValueError(
                "GAN loss (gan_opt), discriminator network (network_d), and discriminator optimizer (optim_d) "
                "must all be enabled or all be disabled.")

        # setup batch augmentations
        self.setup_batchaug()

        # setup gradient clipping
        # self.setup_gradclip(self.net_g, self.net_d)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if self.net_d is not None:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

            # moa
            if self.is_train and self.use_moa:
                self.gt, self.lq = self.batchaugment(self.gt, self.lq)

    def optimize_parameters(self, current_iter):

        # optimize net_g
        if self.net_d is not None:
            for p in self.net_d.parameters():
                p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix
        if self.cri_mssim:
            l_g_mssim = self.cri_mssim(self.output, self.gt)
            l_g_total += l_g_mssim
            loss_dict['l_g_mssim'] = l_g_mssim
        if self.cri_ldl:
            pixel_weight = get_refined_artifact_map(self.gt, self.output, self.net_g_ema(self.lq), 7)
            l_g_ldl = self.cri_ldl(torch.mul(pixel_weight, self.output), torch.mul(pixel_weight, self.gt))
            l_g_total += l_g_ldl
            loss_dict['l_g_ldl'] = l_g_ldl
        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style
        # dists loss
        if self.cri_dists:
            l_g_dists = self.cri_dists(self.output, self.gt)
            l_g_total += l_g_dists
            loss_dict['l_g_dists'] = l_g_dists
        # contextual loss
        if self.cri_contextual:
            l_g_contextual = self.cri_contextual(self.output, self.gt)
            l_g_total += l_g_contextual
            loss_dict['l_g_contextual'] = l_g_contextual
        # color loss
        if self.cri_color:
            l_g_color = self.cri_color(self.output, self.gt)
            l_g_total += l_g_color
            loss_dict['l_g_color'] = l_g_color
        # luma loss
        if self.cri_luma:
            l_g_luma = self.cri_luma(self.output, self.gt)
            l_g_total += l_g_luma
            loss_dict['l_g_luma'] = l_g_luma
        # hsluv loss
        if self.cri_hsluv:
            l_g_hsluv = self.cri_hsluv(self.output, self.gt)
            l_g_total += l_g_hsluv
            loss_dict['l_g_hsluv'] = l_g_hsluv
        # avg loss
        if self.cri_avg:
            l_g_avg = self.cri_avg(self.output, self.gt)
            l_g_total += l_g_avg
            loss_dict['l_g_avg'] = l_g_avg
        # bicubic loss
        if self.cri_bicubic:
            l_g_bicubic = self.cri_bicubic(self.output, self.gt)
            l_g_total += l_g_bicubic
            loss_dict['l_g_bicubic'] = l_g_bicubic
        # gan loss
        if self.cri_gan:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        if self.net_d is not None:
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        self.is_train = False

        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        self.is_train = True

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)

        if self.net_d is not None:
            self.save_network(self.net_d, 'net_d', current_iter)

        self.save_training_state(epoch, current_iter)
