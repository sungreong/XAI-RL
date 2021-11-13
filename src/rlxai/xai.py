from matplotlib import use
from rlxai.img_utils import Rescale, RandomCrop, ToTensor, read_img
from torchvision import transforms
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import torch.nn.functional as F
import torch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum.attr import Lime, LimeBase
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
import matplotlib.pyplot as plt
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import GuidedGradCam, GuidedBackprop, InputXGradient
import math
import warnings
import os

warnings.filterwarnings("ignore")
data_transform = transforms.Compose([Rescale(256), ToTensor()])  #


def show_attr(attr_map, save_path):
    plt_fig, _ = viz.visualize_image_attr(
        attr_map.permute(1, 2, 0).numpy(),  # adjust shape to height, width, channels
        method="heat_map",
        sign="all",
        show_colorbar=True,
        use_pyplot=False,
    )
    if save_path is None:
        plt.show()
    else:
        plt_fig.savefig(save_path)
        plt.close()


class ImageXAI(object):
    def __init__(self, model, target2idx, data_transform, seed=1234):
        self.target2idx = target2idx
        self.idx2target = {v: k for k, v in target2idx.items()}
        self.model = model
        self.data_transform = data_transform
        self.model.eval()
        self.seed = seed

    def get_model(
        self,
    ):
        print(self.model)
        return self.model

    def showImg(self, save_path):
        plt.imshow(self.input_dict["image"])
        plt.title(f"Action : {self.pred_class_eng}")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def showResult(self, save_path):
        self.showImg(save_path)
        print(self.idx2target)
        print("probability : ", self.probs)

    def make_input(self, x, action):
        return {"image": x, "action": action}

    def transform_input(self, input_dict):
        return self.data_transform(input_dict["image"])

    def change_dim(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(dim=0)
        return x

    def __call__(self, x, action):
        print("run...")
        self.input_dict = self.make_input(x, action)
        print("classify...")
        self.pred_class_idx, self.prediction_score, self.real_class_idx, self.probs = self.classify(self.input_dict)
        self.real_class_eng = self.idx2target[self.real_class_idx]
        self.pred_class_eng = self.idx2target[self.pred_class_idx]
        print("run for xai...")
        self.run_IntegratedGradients()
        self.run_GradientShap()
        self.run_Occlusion()
        self.run_LRLIME()
        # self.run_LASSO_LIME()

    def run_LRLIME(
        self,
    ):
        exp_eucl_distance = get_exp_kernel_similarity_function("euclidean", kernel_width=1000)
        self.lr_lime = Lime(
            self.model,
            interpretable_model=SkLearnLinearRegression(),  # build-in wrapped sklearn Linear Regression
            similarity_func=exp_eucl_distance,
        )

    def __run_LASSO_LIME(
        self,
    ):
        n_interpret_features = len(self.target2idx)
        exp_eucl_distance = get_exp_kernel_similarity_function("euclidean", kernel_width=1000)

        def iter_combinations(*args, **kwargs):
            for i in range(2 ** n_interpret_features):
                yield torch.tensor([int(d) for d in bin(i)[2:].zfill(n_interpret_features)]).unsqueeze(0)

        self.lasso_lime = Lime(
            self.model,
            interpretable_model=SkLearnLasso(alpha=0.08),
            similarity_func=exp_eucl_distance,
            perturb_func=iter_combinations,
        )

    def __plot_LASSO_LIME(
        self,
    ):
        n_interpret_features = len(self.target2idx)
        input = self.change_dim(self.transform_input(self.input_dict))
        attrs = self.lasso_lime.attribute(
            input,
            target=self.real_class_idx,
            feature_mask=None,
            n_samples=2 ** n_interpret_features,
            perturbations_per_eval=16,
            show_progress=True,
        ).squeeze(0)
        show_attr(attrs)

    def run_IntegratedGradients(
        self,
    ):
        self.integrated_gradients = IntegratedGradients(self.model)
        input = self.change_dim(self.transform_input(self.input_dict))
        self.attributions_ig = self.integrated_gradients.attribute(input, target=self.pred_class_idx, n_steps=200)
        attr_ig, delta = self.attribute_image_features(
            self.integrated_gradients, input, baselines=input * 0, return_convergence_delta=True
        )
        self.attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print("Approximation delta: ", abs(delta))

    def run_GradientShap(
        self,
    ):
        self.gradient_shap = GradientShap(self.model)

    def run_Occlusion(
        self,
    ):
        self.occlusion = Occlusion(self.model)

    def plot_InputXGradient(self, target, save_path):
        input_x_gradient = InputXGradient(self.model)
        input = self.change_dim(self.transform_input(self.input_dict))
        attribution = input_x_gradient.attribute(input, target)
        original_image = self.input_dict["image"]
        attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        plt_fig, _ = viz.visualize_image_attr(
            attribution,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title=f"Overlayed InputXGradient target : {self.idx2target[target]}",
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_InputXGradient_all_target(self, n_row=2, figsize=(30, 30), save_path=None):
        input_x_gradient = InputXGradient(self.model)
        input = self.change_dim(self.transform_input(self.input_dict))
        total_n = len(self.idx2target)
        n_col = math.ceil(total_n / n_row)
        # plt.figure()
        fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.01, hspace=0.1)
        axes = ax.flatten()
        for idx, (k, v) in enumerate(self.idx2target.items()):
            attribution = input_x_gradient.attribute(input, k)
            original_image = self.input_dict["image"]
            attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            _ = viz.visualize_image_attr(
                attribution,
                original_image,
                method="blended_heat_map",
                sign="absolute_value",
                plt_fig_axis=[fig, axes[idx]],
                outlier_perc=10,
                show_colorbar=False,
                title=f"Overlayed InputXGradient target : {v}",
                use_pyplot=False,
            )
        for remain_idx in range(idx, len(axes)):
            axes[remain_idx].set_axis_off()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def plot_GuidedBackprop(self, target, save_path=None):
        gbp = GuidedBackprop(self.model)
        input = self.change_dim(self.transform_input(self.input_dict))
        attribution = gbp.attribute(input, target)
        original_image = self.input_dict["image"]
        attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        plt_fig, _ = viz.visualize_image_attr(
            attribution,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title=f"Overlayed GuidedBackprop target : {self.idx2target[target]}",
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_GuidedBackprop_all_target(self, n_row=2, figsize=(30, 30), save_path=None):
        gbp = GuidedBackprop(self.model)
        input = self.change_dim(self.transform_input(self.input_dict))
        total_n = len(self.idx2target)
        n_col = math.ceil(total_n / n_row)
        # plt.figure()
        fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.01, hspace=0.1)
        axes = ax.flatten()
        for idx, (k, v) in enumerate(self.idx2target.items()):
            attribution = gbp.attribute(input, k)
            original_image = self.input_dict["image"]
            attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            _ = viz.visualize_image_attr(
                attribution,
                original_image,
                method="blended_heat_map",
                sign="absolute_value",
                plt_fig_axis=[fig, axes[idx]],
                outlier_perc=10,
                show_colorbar=False,
                title=f"Overlayed GuidedBackprop target : {v}",
                use_pyplot=False,
            )
        for remain_idx in range(idx, len(axes)):
            axes[remain_idx].set_axis_off()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def plot_GuidedGradCam(self, model_layer, target, save_path=None):
        guided_gc = GuidedGradCam(self.model, model_layer)
        input = self.change_dim(self.transform_input(self.input_dict))
        attribution = guided_gc.attribute(input, target)
        original_image = self.input_dict["image"]
        attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        plt_fig, _ = viz.visualize_image_attr(
            attribution,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title=f"Overlayed GuidedGradCam target : {self.idx2target[target]}",
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_GuidedGradCam_all_target(self, model_layer, n_row=2, figsize=(30, 30), save_path=None):
        guided_gc = GuidedGradCam(self.model, model_layer)
        input = self.change_dim(self.transform_input(self.input_dict))
        total_n = len(self.idx2target)
        n_col = math.ceil(total_n / n_row)
        # plt.figure()
        fig, ax = plt.subplots(n_row, n_col, figsize=figsize)
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, wspace=0.01, hspace=0.1)
        axes = ax.flatten()
        for idx, (k, v) in enumerate(self.idx2target.items()):
            attribution = guided_gc.attribute(input, k)
            original_image = self.input_dict["image"]
            attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
            _ = viz.visualize_image_attr(
                attribution,
                original_image,
                method="blended_heat_map",
                sign="absolute_value",
                plt_fig_axis=[fig, axes[idx]],
                outlier_perc=10,
                show_colorbar=False,
                title=f"Overlayed GuidedGradCam target : {v}",
                use_pyplot=False,
            )
        for remain_idx in range(idx, len(axes)):
            axes[remain_idx].set_axis_off()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def plot_LRLIME(self, save_path):
        input = self.change_dim(self.transform_input(self.input_dict))
        #
        attrs = self.lr_lime.attribute(
            input,
            target=self.real_class_idx,
            feature_mask=None,
            n_samples=40,
            perturbations_per_eval=16,
            show_progress=True,
        ).squeeze(0)
        show_attr(attrs, save_path)

    def plot_IntegratedGradients_Black(self, save_path):
        default_cmap = LinearSegmentedColormap.from_list(
            "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
        )
        plt_fig, _ = viz.visualize_image_attr(
            np.transpose(self.attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(self.transform_input(self.input_dict).cpu().detach().numpy(), (1, 2, 0)),
            method="heat_map",
            cmap=default_cmap,
            show_colorbar=True,
            sign="positive",
            outlier_perc=1,
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_IntegratedGradients_NoiseTunnel(self, save_path):
        default_cmap = LinearSegmentedColormap.from_list(
            "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
        )
        noise_tunnel = NoiseTunnel(self.integrated_gradients)
        input = self.change_dim(self.transform_input(self.input_dict))
        attributions_ig_nt = noise_tunnel.attribute(
            input, nt_samples=10, nt_type="smoothgrad_sq", target=self.pred_class_idx
        )

        plt_fig, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            cmap=default_cmap,
            show_colorbar=True,
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_GradientShap(self, save_path):
        default_cmap = LinearSegmentedColormap.from_list(
            "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
        )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        input = self.change_dim(self.transform_input(self.input_dict))

        rand_img_dist = torch.cat([input * 0, input * 1])

        attributions_gs = self.gradient_shap.attribute(
            input, n_samples=50, stdevs=0.0001, baselines=rand_img_dist, target=self.pred_class_idx
        )
        plt_fig, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "absolute_value"],
            cmap=default_cmap,
            show_colorbar=True,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def plot_Occlusion(self, strides=(3, 8, 8), sliding_window_shapes=(3, 15, 15), save_path=None):
        input = self.change_dim(self.transform_input(self.input_dict))
        attributions_occ = self.occlusion.attribute(
            input,
            strides=strides,
            target=self.pred_class_idx,
            sliding_window_shapes=sliding_window_shapes,
            baselines=0,
        )
        plt_fig, _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(input.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map"],
            ["all", "positive"],
            show_colorbar=True,
            outlier_perc=2,
            use_pyplot=False,
        )
        if save_path is None:
            plt.show()
        else:
            plt_fig.savefig(save_path)
            plt.close()

    def classify(self, input_dict):
        input = self.change_dim(self.transform_input(input_dict))
        output = self.model(input)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        prediction_score.squeeze_()
        pred_label_idx.squeeze_()
        return pred_label_idx.item(), prediction_score.item(), int(input_dict["action"]), output

    def attribute_image_features(self, algorithm, input, **kwargs):
        self.model.zero_grad()
        tensor_attributions = algorithm.attribute(input, target=self.real_class_idx, **kwargs)

        return tensor_attributions

    def show_encoded_img(self, save_path):
        input = self.change_dim(self.transform_input(self.input_dict))
        result = self.model.ae.encoder(input)
        result = result.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 9))
        ax1.imshow(self.input_dict["image"])
        ax1.set_title(f"input image")
        ax2.imshow(result)
        ax2.set_title(f"img class : {self.real_class_eng}")
        ax3.pcolor(result.sum(axis=2))
        ax3.set_title(f"heatmap class : {self.real_class_eng}")
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    def show_integrated_gradients(self, save_folder):
        input = self.change_dim(self.transform_input(self.input_dict))
        print("Predicted:", self.pred_class_idx, " Probability:", self.probs)
        saliency = Saliency(self.model)
        grads = saliency.attribute(input, target=self.real_class_idx)
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
        nt = NoiseTunnel(self.integrated_gradients)
        attr_ig_nt = self.attribute_image_features(
            nt, input, baselines=input * 0, nt_type="smoothgrad_sq", nt_samples=100, stdevs=0.2
        )
        attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        # dl = DeepLift(self.model)
        # attr_dl = self.attribute_image_features(dl, input, baselines=input * 0)
        # attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        original_image = self.input_dict["image"]
        print("step1")
        plt_fig, _ = viz.visualize_image_attr(
            None, original_image, method="original_image", title="Original Image", use_pyplot=False
        )
        if save_folder is None:
            plt.show()
        else:
            plt_fig.savefig(os.path.join(save_folder, "Orignal_Image.png"), bbox_inches="tight")
        print("step2")
        plt_fig, _ = viz.visualize_image_attr(
            grads,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title="Overlayed Gradient Magnitudes",
            use_pyplot=False,
        )
        if save_folder is None:
            plt.show()
        else:
            plt_fig.savefig(os.path.join(save_folder, "Overlayed_Gradient_Magnitudes.png"), bbox_inches="tight")
        print("step3")
        plt_fig, _ = viz.visualize_image_attr(
            self.attr_ig,
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed Integrated Gradients",
            use_pyplot=False,
        )
        if save_folder is None:
            plt.show()
        else:
            plt_fig.savefig(os.path.join(save_folder, "Overlayed_Integrated_Gradients.png"), bbox_inches="tight")
        print("step4")
        plt_fig, _ = viz.visualize_image_attr(
            attr_ig_nt,
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title="Overlayed Integrated Gradients \n with SmoothGrad Squared",
            use_pyplot=False,
        )
        if save_folder is None:
            plt.show()
        else:
            plt_fig.savefig(
                os.path.join(save_folder, "Overlayed_Integrated_Gradients_with_SmoothGrad_Squared.png"),
                bbox_inches="tight",
            )

        # _ = viz.visualize_image_attr(
        #     attr_dl,
        #     original_image,
        #     method="blended_heat_map",
        #     sign="all",
        #     show_colorbar=True,
        #     title="Overlayed DeepLift",
        # )
