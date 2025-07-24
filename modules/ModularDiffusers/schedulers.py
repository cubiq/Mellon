import logging

from diffusers import ComponentSpec

from mellon.NodeBase import NodeBase

from . import components


logger = logging.getLogger("mellon")


def _create_algorithm_config(options, default="dpmsolver++"):
    """Create algorithm type configuration with given options."""
    return {
        "algorithm_type": {
            "label": "Algorithm Type",
            "type": "string",
            "options": options,
            "value": default,
        }
    }


def _create_solver_order_config(default_order=2, min_order=2, max_order=3):
    """Create solver order configuration."""
    return {
        "solver_order": {
            "label": "Solver Order",
            "type": "int",
            "value": default_order,
            "min": min_order,
            "max": max_order,
        }
    }


ALGORITHM_OPTIONS = {
    "MULTISTEP": {
        "dpmsolver++": "dpm++",
        "sde-dpmsolver++": "sde++",
        "dpmsolver": "dpm",
        "sde-dpmsolver": "sde",
    },
    "SINGLESTEP": {
        "dpmsolver++": "dpm++",
        "sde-dpmsolver++": "sde++",
    },
}


SIGMAS_PARAM = {
    "sigmas": {
        "label": "Sigmas",
        "type": "string",
        "options": {
            "default": "Default",
            "use_karras_sigmas": "Karras",
            "use_exponential_sigmas": "Exponential",
            "use_beta_sigmas": "Beta",
        },
        "value": "default",
    },
}

TIMESTEP_SPACING_PARAM = {
    "timestep_spacing": {
        "label": "Timestep Spacing",
        "type": "string",
        "options": {
            "leading": "Leading",
            "trailing": "Trailing",
            "linspace": "Linspace",
        },
        "value": "leading",
    },
}

RESCALE_BETAS_PARAM = {
    "rescale_betas_zero_snr": {
        "label": "Rescale Betas Zero Snr",
        "type": "boolean",
        "value": False,
    },
}

FINAL_SIGMAS_TYPE_PARAM = {
    "final_sigmas_type": {
        "label": "Final Sigmas Type",
        "type": "string",
        "options": {
            "zero": "Zero",
            "sigma_min": "Last Sigma",
        },
        "value": "zero",
    },
}

SOLVER_TYPE_PARAM = {
    "solver_type": {
        "label": "2nd Order Solver Type",
        "type": "string",
        "options": {
            "midpoint": "Midpoint",
            "heun": "Heun",
        },
        "value": "midpoint",
    },
}

CLIP_SAMPLE_PARAM = {
    "clip_sample": {
        "label": "Clip Sample",
        "type": "boolean",
        "value": False,
    },
    "clip_sample_range": {
        "label": "Clip Sample Range",
        "type": "float",
        "value": 1.0,
        "min": 1.0,
        "max": 10.0,
    },
}

BETA_SCHEDULER_PARAM = {
    "beta_schedule": {
        "label": "Beta Scheduler",
        "type": "string",
        "options": {
            "scaled_linear": "Scaled Linear",
            "linear": "Linear",
            "squaredcos_cap_v2": "Glide Cosine",
        },
        "value": "scaled_linear",
    },
}

TIMESTEP_SCALING_PARAM = {
    "timestep_scaling": {
        "label": "Timestep Scaling",
        "type": "float",
        "value": 10.0,
        "min": 0.0,
        "max": 100.0,
    },
}

LOWER_ORDER_FINAL_PARAM = {
    "lower_order_final": {
        "label": "Use lower-order solvers",
        "type": "boolean",
        "value": True,
    },
}

# Scheduler configurations
SCHEDULER_CONFIGS = {
    "DDIMScheduler": {
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "DDPMScheduler": {
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "DEISMultistepScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "DPMSolverMultistepScheduler": {
        **_create_algorithm_config(ALGORITHM_OPTIONS["MULTISTEP"]),
        **SOLVER_TYPE_PARAM,
        "euler_at_final": {
            "label": "Euler At Final",
            "type": "boolean",
            "value": False,
        },
        **SIGMAS_PARAM,
        "use_lu_lambdas": {
            "label": "Use Lu Lambdas",
            "type": "boolean",
            "value": False,
        },
        **FINAL_SIGMAS_TYPE_PARAM,
        **TIMESTEP_SPACING_PARAM,
        **_create_solver_order_config(default_order=2),
        **RESCALE_BETAS_PARAM,
    },
    "DPMSolverSinglestepScheduler": {
        **_create_algorithm_config(ALGORITHM_OPTIONS["SINGLESTEP"]),
        **SOLVER_TYPE_PARAM,
        **SIGMAS_PARAM,
        **FINAL_SIGMAS_TYPE_PARAM,
        **LOWER_ORDER_FINAL_PARAM,
    },
    "DPMSolverSDEScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
        "noise_sampler_seed": {
            "label": "Noise Sampler Seed",
            "type": "int",
            "value": 0,
            "hidden": True,
        },
    },
    "EulerDiscreteScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
        **FINAL_SIGMAS_TYPE_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "EulerAncestralDiscreteScheduler": {
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "HeunDiscreteScheduler": {
        "beta_schedule": {
            "label": "Beta Scheduler",
            "type": "string",
            "options": {
                "scaled_linear": "Scaled Linear",
                "linear": "Linear",
            },
            "value": "scaled_linear",
        },
        **CLIP_SAMPLE_PARAM,
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "KDPM2DiscreteScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
    },
    "KDPM2AncestralDiscreteScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
    },
    "LCMScheduler": {
        **BETA_SCHEDULER_PARAM,
        **CLIP_SAMPLE_PARAM,
        "set_alpha_to_one": {
            "label": "Alpha to One",
            "type": "boolean",
            "value": True,
        },
        **TIMESTEP_SCALING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "LMSDiscreteScheduler": {
        **SIGMAS_PARAM,
        **TIMESTEP_SPACING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "PNDMScheduler": {
        "skip_prk_steps": {
            "label": "Skip Runge-Kutta steps",
            "type": "boolean",
            "value": False,
        },
        "set_alpha_to_one": {
            "label": "Alpha to One",
            "type": "boolean",
            "value": False,
        },
        **TIMESTEP_SPACING_PARAM,
    },
    "TCDScheduler": {
        **BETA_SCHEDULER_PARAM,
        **CLIP_SAMPLE_PARAM,
        "set_alpha_to_one": {
            "label": "Alpha to One",
            "type": "boolean",
            "value": True,
        },
        **TIMESTEP_SCALING_PARAM,
        **RESCALE_BETAS_PARAM,
    },
    "UniPCMultistepScheduler": {
        **LOWER_ORDER_FINAL_PARAM,
        **SIGMAS_PARAM,
        **FINAL_SIGMAS_TYPE_PARAM,
        **_create_solver_order_config(default_order=3),
        **RESCALE_BETAS_PARAM,
    },
}


class Scheduler(NodeBase):
    label = "Scheduler"
    category = "sampler"
    resizable = True
    skipParamsCheck = True
    params = {
        "scheduler_in": {"label": "Scheduler", "display": "input", "type": "scheduler"},
        "scheduler": {
            "label": "Scheduler",
            "fieldOptions": {"loading": True},
            "options": {
                "DDIMScheduler": "dimm",
                "DDPMScheduler": "ddpm",
                "DEISMultistepScheduler": "deis",
                "DPMSolverMultistepScheduler": "dpmpp_2m",
                "DPMSolverSinglestepScheduler": "dpmpp_2s",
                "DPMSolverSDEScheduler": "dpmpp_sde",
                "EulerDiscreteScheduler": "euler",
                "EulerAncestralDiscreteScheduler": "euler ancestral",
                "HeunDiscreteScheduler": "heun",
                "KDPM2DiscreteScheduler": "kdpm2",
                "KDPM2AncestralDiscreteScheduler": "kdpm2 ancestral",
                "LCMScheduler": "lcm",
                "LMSDiscreteScheduler": "lms",
                "PNDMScheduler": "pndm",
                "TCDScheduler": "tcd",
                "UniPCMultistepScheduler": "unipc",
            },
            "value": "EulerDiscreteScheduler",
            "onChange": "updateNode",
        },
        "scheduler_out": {"label": "Scheduler", "display": "output", "type": "scheduler"},
        "prediction_type": {
            "label": "Prediction Type",
            "type": "string",
            "options": {
                "epsilon": "Epsilon",
                "v_prediction": "V Prediction",
            },
            "value": "epsilon",
        },
    }

    def updateNode(self, values, ref):
        value = values.get("scheduler")
        key = ref.get("key")
        value_from_key = values.get(key)

        print(f"value: {value}, key: {key}, value_from_key: {value_from_key}")
        print("All values", values)

        params = SCHEDULER_CONFIGS.get(value, {})
        self.send_node_definition(params)

    def execute(self, scheduler_in, scheduler, **kwargs):
        logger.debug(f" Scheduler ({self.node_id}) received parameters:")
        logger.debug(f" - input_scheduler: {scheduler_in}")
        logger.debug(f" - scheduler: {scheduler}")
        logger.debug(f" - kwargs: {kwargs}")

        scheduler_component = components.get_one(scheduler_in["model_id"])
        scheduler_cls = getattr(__import__("diffusers", fromlist=[scheduler]), scheduler)

        scheduler_options = {}
        for key, value in kwargs.items():
            if key == "sigmas":
                if value != "default":
                    scheduler_options[value] = True
            elif key == "solver_order":
                scheduler_options[key] = int(value)
            else:
                scheduler_options[key] = value

        logger.debug(f" - scheduler_options: {scheduler_options}")

        schedule_spec = ComponentSpec(
            name="scheduler",
            type_hint=scheduler_cls,
            config=scheduler_component.config,
            default_creation_method="from_config",
        )
        new_scheduler = schedule_spec.create(**scheduler_options)
        comp_id = components.add(name="scheduler", component=new_scheduler, collection=self.node_id)
        logger.debug(f" Scheduler: new_scheduler: {new_scheduler}")

        return {"scheduler_out": components.get_model_info(comp_id)}
